#include "FiniteDifferenceSolver.H"

#include "Utils/TextMsg.H"
#include "Utils/WarpXAlgorithmSelection.H"
#include "Utils/WarpXConst.H"

#include <AMReX_Array4.H>
#include <AMReX_Box.H>
#include <AMReX_MFIter.H>
#include <AMReX_MultiFab.H>
#include <AMReX_REAL.H>

#include <algorithm>
#include <array>
#include <memory>
#include <vector>

using namespace amrex;

namespace
{
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
    Real const c = PhysConst::c;
    Real const dtd2 = 0.5_rt * dt;
    Real const inv_dx = 1._rt / dx;
    Real const inv_dy = 1._rt / dy;
    Real const inv_dz = 1._rt / dz;
    Real const cx = 4._rt * dx * dx / (c * c * dt * dt);
    Real const cy = 4._rt * dy * dy / (c * c * dt * dt);
    Real const cz = 4._rt * dz * dz / (c * c * dt * dt);
    Real const diag_x = 2._rt + cx;
    Real const diag_y = 2._rt + cy;
    Real const diag_z = 2._rt + cz;

    fill_periodic(Efield, periodicity);
    fill_periodic(Bfield, periodicity);

    MultiFab Ex0 = make_copy(*Efield[0]);
    MultiFab Ey0 = make_copy(*Efield[1]);
    MultiFab Ez0 = make_copy(*Efield[2]);

    MultiFab rhs_x = make_rhs(*Efield[0]);
    MultiFab rhs_y = make_rhs(*Efield[1]);
    MultiFab rhs_z = make_rhs(*Efield[2]);

    for (MFIter mfi(rhs_x); mfi.isValid(); ++mfi) {
        auto const rhs = rhs_x.array(mfi);
        auto const ex = Ex0.const_array(mfi);
        auto const ey = Ey0.const_array(mfi);
        auto const by = Bfield[1]->const_array(mfi);
        auto const bz = Bfield[2]->const_array(mfi);
        Box const& bx = mfi.validbox();
        for (int k = bx.smallEnd(2); k <= bx.bigEnd(2); ++k) {
            for (int j = bx.smallEnd(1); j <= bx.bigEnd(1); ++j) {
                for (int i = bx.smallEnd(0); i <= bx.bigEnd(0); ++i) {
                    rhs(i,j,k) = cy * ex(i,j,k)
                        + 2._rt * dy * dy / dt *
                          ((bz(i,j,k) - bz(i,j-1,k)) * inv_dy -
                           (by(i,j,k) - by(i,j,k-1)) * inv_dz)
                        + dy * ((ey(i+1,j-1,k) - ey(i,j-1,k)) * inv_dx -
                                (ey(i+1,j,k) - ey(i,j,k)) * inv_dx);
                }
            }
        }
    }
    solve_periodic_lines(*Efield[0], rhs_x, 1, diag_y);

    for (MFIter mfi(rhs_y); mfi.isValid(); ++mfi) {
        auto const rhs = rhs_y.array(mfi);
        auto const ey = Ey0.const_array(mfi);
        auto const ez = Ez0.const_array(mfi);
        auto const bx = Bfield[0]->const_array(mfi);
        auto const bz = Bfield[2]->const_array(mfi);
        Box const& b = mfi.validbox();
        for (int k = b.smallEnd(2); k <= b.bigEnd(2); ++k) {
            for (int j = b.smallEnd(1); j <= b.bigEnd(1); ++j) {
                for (int i = b.smallEnd(0); i <= b.bigEnd(0); ++i) {
                    rhs(i,j,k) = cz * ey(i,j,k)
                        + 2._rt * dz * dz / dt *
                          ((bx(i,j,k) - bx(i,j,k-1)) * inv_dz -
                           (bz(i,j,k) - bz(i-1,j,k)) * inv_dx)
                        + dz * ((ez(i,j+1,k-1) - ez(i,j,k-1)) * inv_dy -
                                (ez(i,j+1,k) - ez(i,j,k)) * inv_dy);
                }
            }
        }
    }
    solve_periodic_lines(*Efield[1], rhs_y, 2, diag_z);

    for (MFIter mfi(rhs_z); mfi.isValid(); ++mfi) {
        auto const rhs = rhs_z.array(mfi);
        auto const ez = Ez0.const_array(mfi);
        auto const ex = Ex0.const_array(mfi);
        auto const bx = Bfield[0]->const_array(mfi);
        auto const by = Bfield[1]->const_array(mfi);
        Box const& b = mfi.validbox();
        for (int k = b.smallEnd(2); k <= b.bigEnd(2); ++k) {
            for (int j = b.smallEnd(1); j <= b.bigEnd(1); ++j) {
                for (int i = b.smallEnd(0); i <= b.bigEnd(0); ++i) {
                    rhs(i,j,k) = cx * ez(i,j,k)
                        + 2._rt * dx * dx / dt *
                          ((by(i,j,k) - by(i-1,j,k)) * inv_dx -
                           (bx(i,j,k) - bx(i,j-1,k)) * inv_dy)
                        + dx * ((ex(i-1,j,k+1) - ex(i-1,j,k)) * inv_dz -
                                (ex(i,j,k+1) - ex(i,j,k)) * inv_dz);
                }
            }
        }
    }
    solve_periodic_lines(*Efield[2], rhs_z, 0, diag_x);

    fill_periodic(Efield, periodicity);

    for (MFIter mfi(*Bfield[0]); mfi.isValid(); ++mfi) {
        auto const bx = Bfield[0]->array(mfi);
        auto const ey = Efield[1]->const_array(mfi);
        auto const ez0 = Ez0.const_array(mfi);
        Box const& b = mfi.validbox();
        for (int k = b.smallEnd(2); k <= b.bigEnd(2); ++k)
        for (int j = b.smallEnd(1); j <= b.bigEnd(1); ++j)
        for (int i = b.smallEnd(0); i <= b.bigEnd(0); ++i) {
            bx(i,j,k) += dtd2 * ((ey(i,j,k+1) - ey(i,j,k)) * inv_dz -
                                 (ez0(i,j+1,k) - ez0(i,j,k)) * inv_dy);
        }
    }
    for (MFIter mfi(*Bfield[1]); mfi.isValid(); ++mfi) {
        auto const by = Bfield[1]->array(mfi);
        auto const ez = Efield[2]->const_array(mfi);
        auto const ex0 = Ex0.const_array(mfi);
        Box const& b = mfi.validbox();
        for (int k = b.smallEnd(2); k <= b.bigEnd(2); ++k)
        for (int j = b.smallEnd(1); j <= b.bigEnd(1); ++j)
        for (int i = b.smallEnd(0); i <= b.bigEnd(0); ++i) {
            by(i,j,k) += dtd2 * ((ez(i+1,j,k) - ez(i,j,k)) * inv_dx -
                                 (ex0(i,j,k+1) - ex0(i,j,k)) * inv_dz);
        }
    }
    for (MFIter mfi(*Bfield[2]); mfi.isValid(); ++mfi) {
        auto const bz = Bfield[2]->array(mfi);
        auto const ex = Efield[0]->const_array(mfi);
        auto const ey0 = Ey0.const_array(mfi);
        Box const& b = mfi.validbox();
        for (int k = b.smallEnd(2); k <= b.bigEnd(2); ++k)
        for (int j = b.smallEnd(1); j <= b.bigEnd(1); ++j)
        for (int i = b.smallEnd(0); i <= b.bigEnd(0); ++i) {
            bz(i,j,k) += dtd2 * ((ex(i,j+1,k) - ex(i,j,k)) * inv_dy -
                                 (ey0(i+1,j,k) - ey0(i,j,k)) * inv_dx);
        }
    }

    fill_periodic(Bfield, periodicity);

    MultiFab Exh = make_copy(*Efield[0]);
    MultiFab Eyh = make_copy(*Efield[1]);
    MultiFab Ezh = make_copy(*Efield[2]);
    std::array<MultiFab*, 3> Eh = {&Exh, &Eyh, &Ezh};
    fill_periodic(Eh, periodicity);

    for (MFIter mfi(rhs_x); mfi.isValid(); ++mfi) {
        auto const rhs = rhs_x.array(mfi);
        auto const ex = Exh.const_array(mfi);
        auto const ez = Ezh.const_array(mfi);
        auto const by = Bfield[1]->const_array(mfi);
        auto const bz = Bfield[2]->const_array(mfi);
        Box const& b = mfi.validbox();
        for (int k = b.smallEnd(2); k <= b.bigEnd(2); ++k)
        for (int j = b.smallEnd(1); j <= b.bigEnd(1); ++j)
        for (int i = b.smallEnd(0); i <= b.bigEnd(0); ++i) {
            rhs(i,j,k) = cz * ex(i,j,k)
                + 2._rt * dz * dz / dt *
                  ((bz(i,j,k) - bz(i,j-1,k)) * inv_dy -
                   (by(i,j,k) - by(i,j,k-1)) * inv_dz)
                + dz * ((ez(i+1,j,k-1) - ez(i,j,k-1)) * inv_dx -
                        (ez(i+1,j,k) - ez(i,j,k)) * inv_dx);
        }
    }
    solve_periodic_lines(*Efield[0], rhs_x, 2, diag_z);

    for (MFIter mfi(rhs_y); mfi.isValid(); ++mfi) {
        auto const rhs = rhs_y.array(mfi);
        auto const ey = Eyh.const_array(mfi);
        auto const ex = Exh.const_array(mfi);
        auto const bx = Bfield[0]->const_array(mfi);
        auto const bz = Bfield[2]->const_array(mfi);
        Box const& b = mfi.validbox();
        for (int k = b.smallEnd(2); k <= b.bigEnd(2); ++k)
        for (int j = b.smallEnd(1); j <= b.bigEnd(1); ++j)
        for (int i = b.smallEnd(0); i <= b.bigEnd(0); ++i) {
            rhs(i,j,k) = cx * ey(i,j,k)
                + 2._rt * dx * dx / dt *
                  ((bx(i,j,k) - bx(i,j,k-1)) * inv_dz -
                   (bz(i,j,k) - bz(i-1,j,k)) * inv_dx)
                + dx * ((ex(i-1,j+1,k) - ex(i-1,j,k)) * inv_dy -
                        (ex(i,j+1,k) - ex(i,j,k)) * inv_dy);
        }
    }
    solve_periodic_lines(*Efield[1], rhs_y, 0, diag_x);

    for (MFIter mfi(rhs_z); mfi.isValid(); ++mfi) {
        auto const rhs = rhs_z.array(mfi);
        auto const ez = Ezh.const_array(mfi);
        auto const ey = Eyh.const_array(mfi);
        auto const bx = Bfield[0]->const_array(mfi);
        auto const by = Bfield[1]->const_array(mfi);
        Box const& b = mfi.validbox();
        for (int k = b.smallEnd(2); k <= b.bigEnd(2); ++k)
        for (int j = b.smallEnd(1); j <= b.bigEnd(1); ++j)
        for (int i = b.smallEnd(0); i <= b.bigEnd(0); ++i) {
            rhs(i,j,k) = cy * ez(i,j,k)
                + 2._rt * dy * dy / dt *
                  ((by(i,j,k) - by(i-1,j,k)) * inv_dx -
                   (bx(i,j,k) - bx(i,j-1,k)) * inv_dy)
                + dy * ((ey(i,j-1,k+1) - ey(i,j-1,k)) * inv_dz -
                        (ey(i,j,k+1) - ey(i,j,k)) * inv_dz);
        }
    }
    solve_periodic_lines(*Efield[2], rhs_z, 1, diag_y);

    fill_periodic(Efield, periodicity);

    for (MFIter mfi(*Bfield[0]); mfi.isValid(); ++mfi) {
        auto const bx = Bfield[0]->array(mfi);
        auto const eyh = Eyh.const_array(mfi);
        auto const ez = Efield[2]->const_array(mfi);
        Box const& b = mfi.validbox();
        for (int k = b.smallEnd(2); k <= b.bigEnd(2); ++k)
        for (int j = b.smallEnd(1); j <= b.bigEnd(1); ++j)
        for (int i = b.smallEnd(0); i <= b.bigEnd(0); ++i) {
            bx(i,j,k) += dtd2 * ((eyh(i,j,k+1) - eyh(i,j,k)) * inv_dz -
                                 (ez(i,j+1,k) - ez(i,j,k)) * inv_dy);
        }
    }
    for (MFIter mfi(*Bfield[1]); mfi.isValid(); ++mfi) {
        auto const by = Bfield[1]->array(mfi);
        auto const ezh = Ezh.const_array(mfi);
        auto const ex = Efield[0]->const_array(mfi);
        Box const& b = mfi.validbox();
        for (int k = b.smallEnd(2); k <= b.bigEnd(2); ++k)
        for (int j = b.smallEnd(1); j <= b.bigEnd(1); ++j)
        for (int i = b.smallEnd(0); i <= b.bigEnd(0); ++i) {
            by(i,j,k) += dtd2 * ((ezh(i+1,j,k) - ezh(i,j,k)) * inv_dx -
                                 (ex(i,j,k+1) - ex(i,j,k)) * inv_dz);
        }
    }
    for (MFIter mfi(*Bfield[2]); mfi.isValid(); ++mfi) {
        auto const bz = Bfield[2]->array(mfi);
        auto const exh = Exh.const_array(mfi);
        auto const ey = Efield[1]->const_array(mfi);
        Box const& b = mfi.validbox();
        for (int k = b.smallEnd(2); k <= b.bigEnd(2); ++k)
        for (int j = b.smallEnd(1); j <= b.bigEnd(1); ++j)
        for (int i = b.smallEnd(0); i <= b.bigEnd(0); ++i) {
            bz(i,j,k) += dtd2 * ((exh(i,j+1,k) - exh(i,j,k)) * inv_dy -
                                 (ey(i+1,j,k) - ey(i,j,k)) * inv_dx);
        }
    }

    fill_periodic(Bfield, periodicity);
#endif
}
