#include "AdiPML.H"

#include "Utils/TextMsg.H"
#include "Utils/WarpXAlgorithmSelection.H"
#include "Utils/WarpXConst.H"
#include "WarpX.H"

#include <AMReX_Gpu.H>
#include <AMReX_IntVect.H>
#include <AMReX_MFIter.H>
#include <AMReX_Print.H>

#include <array>
#include <cmath>
#include <string>

using namespace amrex;

namespace
{
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    Real adi_pml_rho (
        Real coord, Real inner_lo, Real inner_hi,
        int do_lo, int do_hi) noexcept
    {
        Real rho = 0._rt;
        if (do_lo && coord < inner_lo) {
            rho = inner_lo - coord;
        }
        if (do_hi && coord > inner_hi) {
            rho = amrex::max(rho, coord - inner_hi);
        }
        return rho;
    }
}

void
WarpX::InitAdiPML ()
{
    adi_pml = 0;
    adi_pml_Lo = IntVect::TheZeroVector();
    adi_pml_Hi = IntVect::TheZeroVector();

#if (AMREX_SPACEDIM != 3)
    return;
#else
    if (macroscopic_time_integrator_algo != MacroscopicTimeSteppingScheme::ADI) {
        return;
    }

    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        if (field_boundary_lo[idim] == FieldBoundaryType::PML) {
            adi_pml_Lo[idim] = 1;
            adi_pml = 1;
        }
        if (field_boundary_hi[idim] == FieldBoundaryType::PML) {
            adi_pml_Hi[idim] = 1;
            adi_pml = 1;
        }
    }
    if (!adi_pml) {
        return;
    }

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        finest_level == 0,
        "Macroscopic ADI PML is implemented only for a single AMR level.");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        do_pml_in_domain != 0,
        "Macroscopic ADI PML is in-domain only. Set warpx.do_pml_in_domain = 1 "
        "so the outer warpx.pml_ncell cells of the existing domain are absorbing. "
        "Extra-box WarpX PML (do_pml_in_domain = 0) is not used by ADI.");

    // Do not construct split-field pml[lev]; FDTD PML stays off.
    do_pml = 0;

    BoxArray const& ba = boxArray(0);
    DistributionMapping const& dm = DistributionMap(0);
    IntVect const ng = IntVect(1);

    for (int idim = 0; idim < 3; ++idim) {
        AllocInitMultiFab(
            adi_pml_stretch[idim], ba, dm, 1, ng,
            "adi_pml_stretch[" + std::to_string(idim) + "]", 1._rt);
        AllocInitMultiFab(
            adi_pml_a[idim], ba, dm, 1, ng,
            "adi_pml_a[" + std::to_string(idim) + "]", 0._rt);
        AllocInitMultiFab(
            adi_pml_b[idim], ba, dm, 1, ng,
            "adi_pml_b[" + std::to_string(idim) + "]", 1._rt);
    }

    auto const& eflag = std::array<IntVect, 3>{
        Efield_fp[0][0]->ixType().toIntVect(),
        Efield_fp[0][1]->ixType().toIntVect(),
        Efield_fp[0][2]->ixType().toIntVect()};
    auto const& bflag = std::array<IntVect, 3>{
        Bfield_fp[0][0]->ixType().toIntVect(),
        Bfield_fp[0][1]->ixType().toIntVect(),
        Bfield_fp[0][2]->ixType().toIntVect()};
    IntVect const& eng = Efield_fp[0][0]->nGrowVect();
    IntVect const& bng = Bfield_fp[0][0]->nGrowVect();

    AllocInitMultiFab(adi_psi_e[AdiPsiE::EXY],
        amrex::convert(ba, eflag[0]), dm, 1, eng, "adi_psi_e[exy]", 0._rt);
    AllocInitMultiFab(adi_psi_e[AdiPsiE::EXZ],
        amrex::convert(ba, eflag[0]), dm, 1, eng, "adi_psi_e[exz]", 0._rt);
    AllocInitMultiFab(adi_psi_e[AdiPsiE::EYX],
        amrex::convert(ba, eflag[1]), dm, 1, eng, "adi_psi_e[eyx]", 0._rt);
    AllocInitMultiFab(adi_psi_e[AdiPsiE::EYZ],
        amrex::convert(ba, eflag[1]), dm, 1, eng, "adi_psi_e[eyz]", 0._rt);
    AllocInitMultiFab(adi_psi_e[AdiPsiE::EZX],
        amrex::convert(ba, eflag[2]), dm, 1, eng, "adi_psi_e[ezx]", 0._rt);
    AllocInitMultiFab(adi_psi_e[AdiPsiE::EZY],
        amrex::convert(ba, eflag[2]), dm, 1, eng, "adi_psi_e[ezy]", 0._rt);

    AllocInitMultiFab(adi_psi_h[AdiPsiH::HXY],
        amrex::convert(ba, bflag[0]), dm, 1, bng, "adi_psi_h[hxy]", 0._rt);
    AllocInitMultiFab(adi_psi_h[AdiPsiH::HXZ],
        amrex::convert(ba, bflag[0]), dm, 1, bng, "adi_psi_h[hxz]", 0._rt);
    AllocInitMultiFab(adi_psi_h[AdiPsiH::HYX],
        amrex::convert(ba, bflag[1]), dm, 1, bng, "adi_psi_h[hyx]", 0._rt);
    AllocInitMultiFab(adi_psi_h[AdiPsiH::HYZ],
        amrex::convert(ba, bflag[1]), dm, 1, bng, "adi_psi_h[hyz]", 0._rt);
    AllocInitMultiFab(adi_psi_h[AdiPsiH::HZX],
        amrex::convert(ba, bflag[2]), dm, 1, bng, "adi_psi_h[hzx]", 0._rt);
    AllocInitMultiFab(adi_psi_h[AdiPsiH::HZY],
        amrex::convert(ba, bflag[2]), dm, 1, bng, "adi_psi_h[hzy]", 0._rt);

    FillAdiPmlProfiles();
    UpdateAdiPmlRecursionCoeffs(dt[0]);

    amrex::Print() << "ADI CFS-PML: in-domain, pml_ncell = " << pml_ncell
                   << ", kappa_max = " << adi_pml_kappa_max
                   << ", alpha_max = " << adi_pml_alpha_max
                   << ", m = " << adi_pml_m << "\n";
#endif
}

void
WarpX::FillAdiPmlProfiles ()
{
    if (!adi_pml) { return; }

    auto const problo = Geom(0).ProbLoArray();
    auto const probhi = Geom(0).ProbHiArray();
    auto const dx = Geom(0).CellSizeArray();
    int const ncell = pml_ncell;
    Real const kappa_max = adi_pml_kappa_max;
    Real const grade_m = adi_pml_m;
    IntVect const do_lo = adi_pml_Lo;
    IntVect const do_hi = adi_pml_Hi;

    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        MultiFab& kappa_mf = *adi_pml_stretch[idim];
        kappa_mf.setVal(1._rt);
        int const idir = idim;
        int const lo_on = do_lo[idim];
        int const hi_on = do_hi[idim];
        if (!lo_on && !hi_on) { continue; }

        for (MFIter mfi(kappa_mf); mfi.isValid(); ++mfi) {
            auto const arr = kappa_mf.array(mfi);
            Box const bx = mfi.fabbox();
            ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                int const idx = (idir == 0) ? i : ((idir == 1) ? j : k);
                Real const coord =
                    problo[idir] + (static_cast<Real>(idx) + 0.5_rt) * dx[idir];
                Real const pml_len = static_cast<Real>(ncell) * dx[idir];
                Real const inner_lo = problo[idir] + pml_len;
                Real const inner_hi = probhi[idir] - pml_len;
                Real const rho = adi_pml_rho(
                    coord, inner_lo, inner_hi, lo_on, hi_on);
                if (rho <= 0._rt || pml_len <= 0._rt) {
                    arr(i,j,k) = 1._rt;
                    return;
                }
                Real frac = rho / pml_len;
                frac = amrex::max(0._rt, amrex::min(1._rt, frac));
                Real const poly = std::pow(frac, grade_m);
                arr(i,j,k) = 1._rt + (kappa_max - 1._rt) * poly;
            });
        }
        adi_pml_stretch[idim]->FillBoundary(Geom(0).periodicity());
        adi_pml_stretch[idim]->setBndry(1._rt);
    }
}

void
WarpX::UpdateAdiPmlRecursionCoeffs (Real a_dt)
{
    if (!adi_pml) { return; }

    Real const dt_half = 0.5_rt * a_dt;
    auto const problo = Geom(0).ProbLoArray();
    auto const probhi = Geom(0).ProbHiArray();
    auto const dx = Geom(0).CellSizeArray();
    int const ncell = pml_ncell;
    Real const alpha_max = adi_pml_alpha_max;
    Real const grade_m = adi_pml_m;
    Real const R = adi_pml_R;
    Real const sigma_max_user = adi_pml_sigma_max;
    IntVect const do_lo = adi_pml_Lo;
    IntVect const do_hi = adi_pml_Hi;
    Real const eta0 = std::sqrt(PhysConst::mu0 / PhysConst::ep0);
    Real const ep0 = PhysConst::ep0;

    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        MultiFab& a_mf = *adi_pml_a[idim];
        MultiFab& b_mf = *adi_pml_b[idim];
        MultiFab const& kappa_mf = *adi_pml_stretch[idim];
        a_mf.setVal(0._rt);
        b_mf.setVal(1._rt);

        int const idir = idim;
        int const lo_on = do_lo[idim];
        int const hi_on = do_hi[idim];
        if (!lo_on && !hi_on) { continue; }

        Real const pml_len = static_cast<Real>(ncell) * dx[idir];
        Real sigma_max = sigma_max_user;
        if (sigma_max < 0._rt && pml_len > 0._rt) {
            sigma_max = -(grade_m + 1._rt) * std::log(R) / (2._rt * eta0 * pml_len);
        }

        for (MFIter mfi(a_mf); mfi.isValid(); ++mfi) {
            auto const a_arr = a_mf.array(mfi);
            auto const b_arr = b_mf.array(mfi);
            auto const k_arr = kappa_mf.const_array(mfi);
            Box const bx = mfi.fabbox();
            ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                int const idx = (idir == 0) ? i : ((idir == 1) ? j : k);
                Real const coord =
                    problo[idir] + (static_cast<Real>(idx) + 0.5_rt) * dx[idir];
                Real const inner_lo = problo[idir] + pml_len;
                Real const inner_hi = probhi[idir] - pml_len;
                Real const rho = adi_pml_rho(
                    coord, inner_lo, inner_hi, lo_on, hi_on);
                if (rho <= 0._rt || pml_len <= 0._rt) {
                    a_arr(i,j,k) = 0._rt;
                    b_arr(i,j,k) = 1._rt;
                    return;
                }
                Real frac = rho / pml_len;
                frac = amrex::max(0._rt, amrex::min(1._rt, frac));
                Real const poly = std::pow(frac, grade_m);
                Real const sig = sigma_max * poly;
                Real const alp = alpha_max * (1._rt - frac);
                Real const kapp = k_arr(i,j,k);
                Real const expo = -((sig / kapp) + alp) * dt_half / ep0;
                Real const bval = std::exp(expo);
                b_arr(i,j,k) = bval;
                Real const denom = kapp * (sig + kapp * alp);
                if (std::abs(denom) > 0._rt) {
                    a_arr(i,j,k) = (sig / denom) * (bval - 1._rt);
                } else {
                    a_arr(i,j,k) = 0._rt;
                }
            });
        }
        adi_pml_a[idim]->FillBoundary(Geom(0).periodicity());
        adi_pml_b[idim]->FillBoundary(Geom(0).periodicity());
        adi_pml_a[idim]->setBndry(0._rt);
        adi_pml_b[idim]->setBndry(1._rt);
    }
}
