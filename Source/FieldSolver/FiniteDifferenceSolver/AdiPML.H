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

#endif
