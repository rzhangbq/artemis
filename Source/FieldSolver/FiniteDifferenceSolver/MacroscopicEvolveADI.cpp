#include "FiniteDifferenceSolver.H"

#include "Utils/TextMsg.H"
#include "Utils/WarpXAlgorithmSelection.H"
#include "Utils/WarpXConst.H"

#include <AMReX_Array4.H>
#include <AMReX_Box.H>
#include <AMReX_MFIter.H>
#include <AMReX_MultiFab.H>
#include <AMReX_REAL.H>

#include <array>
#include <memory>
#include <vector>

using namespace amrex;

namespace
{
    struct AdiCoeffs
    {
        Real dx = 0._rt;
        Real dy = 0._rt;
        Real dz = 0._rt;
        Real inv_dx = 0._rt;
        Real inv_dy = 0._rt;
        Real inv_dz = 0._rt;
        Real cx = 0._rt;
        Real cy = 0._rt;
        Real cz = 0._rt;
        Real diag_x = 0._rt;
        Real diag_y = 0._rt;
        Real diag_z = 0._rt;
        Real dt = 0._rt;
        Real dtd2 = 0._rt;
    };

    AMREX_FORCE_INLINE
    Real line_value (Array4<Real const> const& a, int i, int j, int k,
                     int dir, int p) noexcept
    {
        if (dir == 0) { return a(p, j, k); }
        if (dir == 1) { return a(i, p, k); }
        return a(i, j, p);
    }

    AMREX_FORCE_INLINE
    void set_line_value (Array4<Real> const& a, int i, int j, int k,
                         int dir, int p, Real v) noexcept
    {
        if (dir == 0) {
            a(p, j, k) = v;
        } else if (dir == 1) {
            a(i, p, k) = v;
        } else {
            a(i, j, p) = v;
        }
    }

    void solve_tridiagonal (
        std::vector<Real> const& a,
        std::vector<Real> b,
        std::vector<Real> const& c,
        std::vector<Real>& d)
    {
        int const n = static_cast<int>(d.size());
        for (int i = 1; i < n; ++i) {
            Real const m = a[i] / b[i-1];
            b[i] -= m * c[i-1];
            d[i] -= m * d[i-1];
        }
        d[n-1] /= b[n-1];
        for (int i = n-2; i >= 0; --i) {
            d[i] = (d[i] - c[i] * d[i+1]) / b[i];
        }
    }

    void solve_cyclic_tridiagonal (Real diag, std::vector<Real>& x)
    {
        int const n = static_cast<int>(x.size());
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(n > 2, "ADI cyclic solve needs at least three cells per line.");

        std::vector<Real> a(n, -1._rt);
        std::vector<Real> b(n, diag);
        std::vector<Real> c(n, -1._rt);
        a[0] = 0._rt;
        c[n-1] = 0._rt;

        Real const alpha = -1._rt;
        Real const beta = -1._rt;
        Real const gamma = -diag;

        std::vector<Real> bb = b;
        bb[0] -= gamma;
        bb[n-1] -= alpha * beta / gamma;

        std::vector<Real> u(n, 0._rt);
        u[0] = gamma;
        u[n-1] = alpha;

        std::vector<Real> z = u;
        solve_tridiagonal(a, bb, c, x);
        solve_tridiagonal(a, bb, c, z);

        Real const fact = (x[0] + beta * x[n-1] / gamma) /
                          (1._rt + z[0] + beta * z[n-1] / gamma);
        for (int i = 0; i < n; ++i) {
            x[i] -= fact * z[i];
        }
    }

    void solve_periodic_lines (MultiFab& field, MultiFab const& rhs, int dir, Real diag)
    {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(field.boxArray().size() == 1,
            "Macroscopic ADI currently requires one grid box per field component.");

        for (MFIter mfi(field); mfi.isValid(); ++mfi) {
            Box const& bx = mfi.validbox();
            auto const in = rhs.const_array(mfi);
            auto const out = field.array(mfi);
            IntVect const stag = field.ixType().toIntVect();
            bool const nodal_in_solve_dir = (stag[dir] == 1);

            int const ilo = bx.smallEnd(0);
            int const ihi = bx.bigEnd(0);
            int const jlo = bx.smallEnd(1);
            int const jhi = bx.bigEnd(1);
            int const klo = bx.smallEnd(2);
            int const khi = bx.bigEnd(2);
            int const lo = bx.smallEnd(dir);
            int const hi = bx.bigEnd(dir);
            int const n = nodal_in_solve_dir ? hi - lo : hi - lo + 1;

            std::vector<Real> line(n);
            for (int k = klo; k <= khi; ++k) {
                if (dir == 2 && k != klo) { continue; }
                for (int j = jlo; j <= jhi; ++j) {
                    if (dir == 1 && j != jlo) { continue; }
                    for (int i = ilo; i <= ihi; ++i) {
                        if (dir == 0 && i != ilo) { continue; }
                        for (int p = 0; p < n; ++p) {
                            line[p] = line_value(in, i, j, k, dir, lo + p);
                        }
                        solve_cyclic_tridiagonal(diag, line);
                        for (int p = 0; p < n; ++p) {
                            set_line_value(out, i, j, k, dir, lo + p, line[p]);
                        }
                        if (nodal_in_solve_dir) {
                            set_line_value(out, i, j, k, dir, hi, line[0]);
                        }
                    }
                }
            }
        }
    }

    MultiFab make_rhs (MultiFab const& mf)
    {
        return MultiFab(mf.boxArray(), mf.DistributionMap(), 1, 0);
    }

    MultiFab make_copy (MultiFab const& mf)
    {
        MultiFab copy(mf.boxArray(), mf.DistributionMap(), 1, mf.nGrowVect());
        MultiFab::Copy(copy, mf, 0, 0, 1, mf.nGrowVect());
        return copy;
    }

    void fill_periodic (std::array<std::unique_ptr<MultiFab>, 3>& field,
                        Periodicity const& periodicity)
    {
        for (auto& component : field) {
            component->FillBoundary(periodicity);
        }
    }

    void fill_periodic (std::array<MultiFab*, 3> const& field,
                        Periodicity const& periodicity)
    {
        for (auto* component : field) {
            component->FillBoundary(periodicity);
        }
    }

    // First half-step RHS: implicit Ex along y.
    MultiFab build_rhs_ex1 (
        MultiFab const& ex, MultiFab const& ey,
        MultiFab const& by, MultiFab const& bz,
        AdiCoeffs const& c)
    {
        MultiFab rhs = make_rhs(ex);
        for (MFIter mfi(rhs); mfi.isValid(); ++mfi) {
            auto const rhs_arr = rhs.array(mfi);
            auto const ex_arr = ex.const_array(mfi);
            auto const ey_arr = ey.const_array(mfi);
            auto const by_arr = by.const_array(mfi);
            auto const bz_arr = bz.const_array(mfi);
            Box const& bx = mfi.validbox();
            for (int k = bx.smallEnd(2); k <= bx.bigEnd(2); ++k) {
                for (int j = bx.smallEnd(1); j <= bx.bigEnd(1); ++j) {
                    for (int i = bx.smallEnd(0); i <= bx.bigEnd(0); ++i) {
                        rhs_arr(i,j,k) = c.cy * ex_arr(i,j,k)
                            + 2._rt * c.dy * c.dy / c.dt *
                              ((bz_arr(i,j,k) - bz_arr(i,j-1,k)) * c.inv_dy -
                               (by_arr(i,j,k) - by_arr(i,j,k-1)) * c.inv_dz)
                            + c.dy * ((ey_arr(i+1,j-1,k) - ey_arr(i,j-1,k)) * c.inv_dx -
                                      (ey_arr(i+1,j,k) - ey_arr(i,j,k)) * c.inv_dx);
                    }
                }
            }
        }
        return rhs;
    }

    // First half-step RHS: implicit Ey along z.
    MultiFab build_rhs_ey1 (
        MultiFab const& ey, MultiFab const& ez,
        MultiFab const& bx, MultiFab const& bz,
        AdiCoeffs const& c)
    {
        MultiFab rhs = make_rhs(ey);
        for (MFIter mfi(rhs); mfi.isValid(); ++mfi) {
            auto const rhs_arr = rhs.array(mfi);
            auto const ey_arr = ey.const_array(mfi);
            auto const ez_arr = ez.const_array(mfi);
            auto const bx_arr = bx.const_array(mfi);
            auto const bz_arr = bz.const_array(mfi);
            Box const& b = mfi.validbox();
            for (int k = b.smallEnd(2); k <= b.bigEnd(2); ++k) {
                for (int j = b.smallEnd(1); j <= b.bigEnd(1); ++j) {
                    for (int i = b.smallEnd(0); i <= b.bigEnd(0); ++i) {
                        rhs_arr(i,j,k) = c.cz * ey_arr(i,j,k)
                            + 2._rt * c.dz * c.dz / c.dt *
                              ((bx_arr(i,j,k) - bx_arr(i,j,k-1)) * c.inv_dz -
                               (bz_arr(i,j,k) - bz_arr(i-1,j,k)) * c.inv_dx)
                            + c.dz * ((ez_arr(i,j+1,k-1) - ez_arr(i,j,k-1)) * c.inv_dy -
                                      (ez_arr(i,j+1,k) - ez_arr(i,j,k)) * c.inv_dy);
                    }
                }
            }
        }
        return rhs;
    }

    // First half-step RHS: implicit Ez along x.
    MultiFab build_rhs_ez1 (
        MultiFab const& ez, MultiFab const& ex,
        MultiFab const& bx, MultiFab const& by,
        AdiCoeffs const& c)
    {
        MultiFab rhs = make_rhs(ez);
        for (MFIter mfi(rhs); mfi.isValid(); ++mfi) {
            auto const rhs_arr = rhs.array(mfi);
            auto const ez_arr = ez.const_array(mfi);
            auto const ex_arr = ex.const_array(mfi);
            auto const bx_arr = bx.const_array(mfi);
            auto const by_arr = by.const_array(mfi);
            Box const& b = mfi.validbox();
            for (int k = b.smallEnd(2); k <= b.bigEnd(2); ++k) {
                for (int j = b.smallEnd(1); j <= b.bigEnd(1); ++j) {
                    for (int i = b.smallEnd(0); i <= b.bigEnd(0); ++i) {
                        rhs_arr(i,j,k) = c.cx * ez_arr(i,j,k)
                            + 2._rt * c.dx * c.dx / c.dt *
                              ((by_arr(i,j,k) - by_arr(i-1,j,k)) * c.inv_dx -
                               (bx_arr(i,j,k) - bx_arr(i,j-1,k)) * c.inv_dy)
                            + c.dx * ((ex_arr(i-1,j,k+1) - ex_arr(i-1,j,k)) * c.inv_dz -
                                      (ex_arr(i,j,k+1) - ex_arr(i,j,k)) * c.inv_dz);
                    }
                }
            }
        }
        return rhs;
    }

    // Second half-step RHS: implicit Ex along z.
    MultiFab build_rhs_ex2 (
        MultiFab const& ex, MultiFab const& ez,
        MultiFab const& by, MultiFab const& bz,
        AdiCoeffs const& c)
    {
        MultiFab rhs = make_rhs(ex);
        for (MFIter mfi(rhs); mfi.isValid(); ++mfi) {
            auto const rhs_arr = rhs.array(mfi);
            auto const ex_arr = ex.const_array(mfi);
            auto const ez_arr = ez.const_array(mfi);
            auto const by_arr = by.const_array(mfi);
            auto const bz_arr = bz.const_array(mfi);
            Box const& b = mfi.validbox();
            for (int k = b.smallEnd(2); k <= b.bigEnd(2); ++k) {
                for (int j = b.smallEnd(1); j <= b.bigEnd(1); ++j) {
                    for (int i = b.smallEnd(0); i <= b.bigEnd(0); ++i) {
                        rhs_arr(i,j,k) = c.cz * ex_arr(i,j,k)
                            + 2._rt * c.dz * c.dz / c.dt *
                              ((bz_arr(i,j,k) - bz_arr(i,j-1,k)) * c.inv_dy -
                               (by_arr(i,j,k) - by_arr(i,j,k-1)) * c.inv_dz)
                            + c.dz * ((ez_arr(i+1,j,k-1) - ez_arr(i,j,k-1)) * c.inv_dx -
                                      (ez_arr(i+1,j,k) - ez_arr(i,j,k)) * c.inv_dx);
                    }
                }
            }
        }
        return rhs;
    }

    // Second half-step RHS: implicit Ey along x.
    MultiFab build_rhs_ey2 (
        MultiFab const& ey, MultiFab const& ex,
        MultiFab const& bx, MultiFab const& bz,
        AdiCoeffs const& c)
    {
        MultiFab rhs = make_rhs(ey);
        for (MFIter mfi(rhs); mfi.isValid(); ++mfi) {
            auto const rhs_arr = rhs.array(mfi);
            auto const ey_arr = ey.const_array(mfi);
            auto const ex_arr = ex.const_array(mfi);
            auto const bx_arr = bx.const_array(mfi);
            auto const bz_arr = bz.const_array(mfi);
            Box const& b = mfi.validbox();
            for (int k = b.smallEnd(2); k <= b.bigEnd(2); ++k) {
                for (int j = b.smallEnd(1); j <= b.bigEnd(1); ++j) {
                    for (int i = b.smallEnd(0); i <= b.bigEnd(0); ++i) {
                        rhs_arr(i,j,k) = c.cx * ey_arr(i,j,k)
                            + 2._rt * c.dx * c.dx / c.dt *
                              ((bx_arr(i,j,k) - bx_arr(i,j,k-1)) * c.inv_dz -
                               (bz_arr(i,j,k) - bz_arr(i-1,j,k)) * c.inv_dx)
                            + c.dx * ((ex_arr(i-1,j+1,k) - ex_arr(i-1,j,k)) * c.inv_dy -
                                      (ex_arr(i,j+1,k) - ex_arr(i,j,k)) * c.inv_dy);
                    }
                }
            }
        }
        return rhs;
    }

    // Second half-step RHS: implicit Ez along y.
    MultiFab build_rhs_ez2 (
        MultiFab const& ez, MultiFab const& ey,
        MultiFab const& bx, MultiFab const& by,
        AdiCoeffs const& c)
    {
        MultiFab rhs = make_rhs(ez);
        for (MFIter mfi(rhs); mfi.isValid(); ++mfi) {
            auto const rhs_arr = rhs.array(mfi);
            auto const ez_arr = ez.const_array(mfi);
            auto const ey_arr = ey.const_array(mfi);
            auto const bx_arr = bx.const_array(mfi);
            auto const by_arr = by.const_array(mfi);
            Box const& b = mfi.validbox();
            for (int k = b.smallEnd(2); k <= b.bigEnd(2); ++k) {
                for (int j = b.smallEnd(1); j <= b.bigEnd(1); ++j) {
                    for (int i = b.smallEnd(0); i <= b.bigEnd(0); ++i) {
                        rhs_arr(i,j,k) = c.cy * ez_arr(i,j,k)
                            + 2._rt * c.dy * c.dy / c.dt *
                              ((by_arr(i,j,k) - by_arr(i-1,j,k)) * c.inv_dx -
                               (bx_arr(i,j,k) - bx_arr(i,j-1,k)) * c.inv_dy)
                            + c.dy * ((ey_arr(i,j-1,k+1) - ey_arr(i,j-1,k)) * c.inv_dz -
                                      (ey_arr(i,j,k+1) - ey_arr(i,j,k)) * c.inv_dz);
                    }
                }
            }
        }
        return rhs;
    }

    void solve_implicit_ex1 (MultiFab& ex, MultiFab const& rhs, AdiCoeffs const& c)
    {
        solve_periodic_lines(ex, rhs, 1, c.diag_y);
    }

    void solve_implicit_ey1 (MultiFab& ey, MultiFab const& rhs, AdiCoeffs const& c)
    {
        solve_periodic_lines(ey, rhs, 2, c.diag_z);
    }

    void solve_implicit_ez1 (MultiFab& ez, MultiFab const& rhs, AdiCoeffs const& c)
    {
        solve_periodic_lines(ez, rhs, 0, c.diag_x);
    }

    void solve_implicit_ex2 (MultiFab& ex, MultiFab const& rhs, AdiCoeffs const& c)
    {
        solve_periodic_lines(ex, rhs, 2, c.diag_z);
    }

    void solve_implicit_ey2 (MultiFab& ey, MultiFab const& rhs, AdiCoeffs const& c)
    {
        solve_periodic_lines(ey, rhs, 0, c.diag_x);
    }

    void solve_implicit_ez2 (MultiFab& ez, MultiFab const& rhs, AdiCoeffs const& c)
    {
        solve_periodic_lines(ez, rhs, 1, c.diag_y);
    }

    void step_bx (MultiFab& bx, MultiFab const& ey, MultiFab const& ez, AdiCoeffs const& c)
    {
        for (MFIter mfi(bx); mfi.isValid(); ++mfi) {
            auto const bx_arr = bx.array(mfi);
            auto const ey_arr = ey.const_array(mfi);
            auto const ez_arr = ez.const_array(mfi);
            Box const& b = mfi.validbox();
            for (int k = b.smallEnd(2); k <= b.bigEnd(2); ++k) {
                for (int j = b.smallEnd(1); j <= b.bigEnd(1); ++j) {
                    for (int i = b.smallEnd(0); i <= b.bigEnd(0); ++i) {
                        bx_arr(i,j,k) += c.dtd2 * ((ey_arr(i,j,k+1) - ey_arr(i,j,k)) * c.inv_dz -
                                                   (ez_arr(i,j+1,k) - ez_arr(i,j,k)) * c.inv_dy);
                    }
                }
            }
        }
    }

    void step_by (MultiFab& by, MultiFab const& ez, MultiFab const& ex, AdiCoeffs const& c)
    {
        for (MFIter mfi(by); mfi.isValid(); ++mfi) {
            auto const by_arr = by.array(mfi);
            auto const ez_arr = ez.const_array(mfi);
            auto const ex_arr = ex.const_array(mfi);
            Box const& b = mfi.validbox();
            for (int k = b.smallEnd(2); k <= b.bigEnd(2); ++k) {
                for (int j = b.smallEnd(1); j <= b.bigEnd(1); ++j) {
                    for (int i = b.smallEnd(0); i <= b.bigEnd(0); ++i) {
                        by_arr(i,j,k) += c.dtd2 * ((ez_arr(i+1,j,k) - ez_arr(i,j,k)) * c.inv_dx -
                                                   (ex_arr(i,j,k+1) - ex_arr(i,j,k)) * c.inv_dz);
                    }
                }
            }
        }
    }

    void step_bz (MultiFab& bz, MultiFab const& ex, MultiFab const& ey, AdiCoeffs const& c)
    {
        for (MFIter mfi(bz); mfi.isValid(); ++mfi) {
            auto const bz_arr = bz.array(mfi);
            auto const ex_arr = ex.const_array(mfi);
            auto const ey_arr = ey.const_array(mfi);
            Box const& b = mfi.validbox();
            for (int k = b.smallEnd(2); k <= b.bigEnd(2); ++k) {
                for (int j = b.smallEnd(1); j <= b.bigEnd(1); ++j) {
                    for (int i = b.smallEnd(0); i <= b.bigEnd(0); ++i) {
                        bz_arr(i,j,k) += c.dtd2 * ((ex_arr(i,j+1,k) - ex_arr(i,j,k)) * c.inv_dy -
                                                   (ey_arr(i+1,j,k) - ey_arr(i,j,k)) * c.inv_dx);
                    }
                }
            }
        }
    }

    void adi_first_half_step (
        std::array<std::unique_ptr<MultiFab>, 3>& Efield,
        std::array<std::unique_ptr<MultiFab>, 3>& Bfield,
        AdiCoeffs const& c,
        Periodicity const& periodicity)
    {
        // Implicit E along y,z,x; explicit B at n+1/2.
        MultiFab Ex0 = make_copy(*Efield[0]);
        MultiFab Ey0 = make_copy(*Efield[1]);
        MultiFab Ez0 = make_copy(*Efield[2]);

        MultiFab rhs_ex = build_rhs_ex1(Ex0, Ey0, *Bfield[1], *Bfield[2], c);
        MultiFab rhs_ey = build_rhs_ey1(Ey0, Ez0, *Bfield[0], *Bfield[2], c);
        MultiFab rhs_ez = build_rhs_ez1(Ez0, Ex0, *Bfield[0], *Bfield[1], c);

        solve_implicit_ex1(*Efield[0], rhs_ex, c);
        solve_implicit_ey1(*Efield[1], rhs_ey, c);
        solve_implicit_ez1(*Efield[2], rhs_ez, c);

        fill_periodic(Efield, periodicity);

        step_bx(*Bfield[0], *Efield[1], Ez0, c);
        step_by(*Bfield[1], *Efield[2], Ex0, c);
        step_bz(*Bfield[2], *Efield[0], Ey0, c);

        fill_periodic(Bfield, periodicity);
    }

    void adi_second_half_step (
        std::array<std::unique_ptr<MultiFab>, 3>& Efield,
        std::array<std::unique_ptr<MultiFab>, 3>& Bfield,
        AdiCoeffs const& c,
        Periodicity const& periodicity)
    {
        // Implicit E along z,x,y; explicit B at n+1.
        MultiFab Exh = make_copy(*Efield[0]);
        MultiFab Eyh = make_copy(*Efield[1]);
        MultiFab Ezh = make_copy(*Efield[2]);
        std::array<MultiFab*, 3> Eh = {&Exh, &Eyh, &Ezh};
        fill_periodic(Eh, periodicity);

        MultiFab rhs_ex = build_rhs_ex2(Exh, Ezh, *Bfield[1], *Bfield[2], c);
        MultiFab rhs_ey = build_rhs_ey2(Eyh, Exh, *Bfield[0], *Bfield[2], c);
        MultiFab rhs_ez = build_rhs_ez2(Ezh, Eyh, *Bfield[0], *Bfield[1], c);

        solve_implicit_ex2(*Efield[0], rhs_ex, c);
        solve_implicit_ey2(*Efield[1], rhs_ey, c);
        solve_implicit_ez2(*Efield[2], rhs_ez, c);

        fill_periodic(Efield, periodicity);

        step_bx(*Bfield[0], Eyh, *Efield[2], c);
        step_by(*Bfield[1], Ezh, *Efield[0], c);
        step_bz(*Bfield[2], Exh, *Efield[1], c);

        fill_periodic(Bfield, periodicity);
    }
}

void
FiniteDifferenceSolver::MacroscopicEvolveADI (
    std::array< std::unique_ptr<MultiFab>, 3>& Efield,
    std::array< std::unique_ptr<MultiFab>, 3>& Bfield,
    Real const dt,
    Periodicity const& periodicity)
{
#ifdef WARPX_DIM_RZ
    amrex::ignore_unused(Efield, Bfield, dt, periodicity);
    WARPX_ABORT_WITH_MESSAGE("Macroscopic ADI is implemented only for 3D Cartesian grids.");
#else
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_fdtd_algo == ElectromagneticSolverAlgo::Yee,
        "Macroscopic ADI currently supports only the Yee solver.");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_grid_type == GridType::Staggered,
        "Macroscopic ADI currently supports only staggered Yee fields.");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        periodicity.isPeriodic(0) && periodicity.isPeriodic(1) && periodicity.isPeriodic(2),
        "Macroscopic ADI currently supports only periodic boundaries.");
    for (int n = 0; n < 3; ++n) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(Efield[n]->boxArray().size() == 1 &&
                                         Bfield[n]->boxArray().size() == 1,
            "Macroscopic ADI currently requires one grid box per field component.");
    }

    Real const dx = 1._rt / m_h_stencil_coefs_x[0];
    Real const dy = 1._rt / m_h_stencil_coefs_y[0];
    Real const dz = 1._rt / m_h_stencil_coefs_z[0];
    Real const c0 = PhysConst::c;

    AdiCoeffs c;
    c.dx = dx;
    c.dy = dy;
    c.dz = dz;
    c.inv_dx = 1._rt / dx;
    c.inv_dy = 1._rt / dy;
    c.inv_dz = 1._rt / dz;
    c.dt = dt;
    c.dtd2 = 0.5_rt * dt;
    c.cx = 4._rt * dx * dx / (c0 * c0 * dt * dt);
    c.cy = 4._rt * dy * dy / (c0 * c0 * dt * dt);
    c.cz = 4._rt * dz * dz / (c0 * c0 * dt * dt);
    c.diag_x = 2._rt + c.cx;
    c.diag_y = 2._rt + c.cy;
    c.diag_z = 2._rt + c.cz;

    fill_periodic(Efield, periodicity);
    fill_periodic(Bfield, periodicity);

    adi_first_half_step(Efield, Bfield, c, periodicity);
    adi_second_half_step(Efield, Bfield, c, periodicity);
#endif
}
