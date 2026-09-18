/* Convolutional (CFS) PML auxiliaries for the macroscopic ADI Maxwell solver.
 *
 * These are unsplit CPML fields on the existing ADI domain
 * (warpx.do_pml_in_domain = 1). They are not the split-field WarpX pml[lev]
 * boxes used by explicit FDTD.
 */
#ifndef WARPX_ADI_PML_H_
#define WARPX_ADI_PML_H_

namespace AdiPsiE
{
    enum { EXY = 0, EXZ, EYX, EYZ, EZX, EZY };
}

namespace AdiPsiH
{
    enum { HXY = 0, HXZ, HYX, HYZ, HZX, HZY };
}

namespace amrex { class MultiFab; }

// quantity: 0 = kappa, 1 = a, 2 = b. Fills dst from physical coordinates
// using dst's IndexType (cell or node) along `dir`.
void FillAdiCfsOnLayout (amrex::MultiFab& dst, int dir, int quantity);

#endif
