import numpy as np
from elitho import diffraction_amplitude, const, descriptors, diffraction_order

def reflect_amplitude():
    dod_narrow = descriptors.DiffractionOrderDescriptor(1.5)
    doc_narrow = diffraction_order.DiffractionOrderCoordinate(
        dod_narrow.max_diffraction_order_x,
        dod_narrow.max_diffraction_order_y,
        diffraction_order.ellipse,
    )
    abxx, vcxx = diffraction_amplitude.absorber_and_vacuum_amplitudes(
        const.PolarizationDirection.X, dod_narrow, doc_narrow
    )
    ampab = abxx[const.lsmaxX, const.lsmaxY]
    ampvc = vcxx[const.lsmaxX, const.lsmaxY]
    phase0 = ampvc/np.abs(ampvc)
    return ampvc, ampab, phase0

if __name__ == "__main__":
    ampvc,ampab,phase0 = reflect_amplitude()
    print(ampvc,ampab,phase0)
