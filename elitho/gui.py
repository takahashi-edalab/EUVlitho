import json
import numpy as np
import matplotlib.pyplot as plt
from elitho import config
import streamlit as st

# Ensure project root is on sys.path so `elitho` can be imported when running
# this file directly (for example with `streamlit run elitho/gui.py`).
import os
import sys

try:
    from elitho import illumination, source
except Exception:
    _here = os.path.dirname(__file__)
    _root = os.path.abspath(os.path.join(_here, os.pardir))
    if _root not in sys.path:
        sys.path.insert(0, _root)
    from elitho import illumination, source


def simulate_psf(wl_nm, NA, img_size=512, pixels_per_um=50, sigma_factor=1.0):
    """Approximate PSF: gaussian-like disk scaled by sigma_factor.

    Returns (img_normalized, airy_radius_um)
    """
    wl_um = wl_nm / 1000.0
    NA = max(1e-6, float(NA))
    r_um = 1.22 * wl_um / NA
    sigma_um = max(1e-6, r_um * sigma_factor)
    half_size_um = img_size / (2.0 * pixels_per_um)
    x = np.linspace(-half_size_um, half_size_um, img_size)
    xv, yv = np.meshgrid(x, x)
    rr2 = xv**2 + yv**2
    sigma_pixels = sigma_um * pixels_per_um
    sigma_pixels = max(1e-6, sigma_pixels)
    img = np.exp(-rr2 / (2.0 * (sigma_pixels**2)))
    img = img / img.max()
    return img, r_um


def generate_mask(width, height, opens, pixels_per_um=10):
    width = max(1e-6, float(width))
    height = max(1e-6, float(height))

    width_px = max(1, int(round(width * pixels_per_um)))
    height_px = max(1, int(round(height * pixels_per_um)))
    mask = np.zeros((height_px, width_px), dtype=np.uint8)

    for op in opens:
        cx = float(op.get("center_x_um", 0.0))
        cy = float(op.get("center_y_um", 0.0))
        w = float(op.get("width_um", 0.0))
        h = float(op.get("height_um", 0.0))

        x0 = int(round((cx + width / 2.0 - w / 2.0) * pixels_per_um))
        x1 = int(round((cx + width / 2.0 + w / 2.0) * pixels_per_um))
        y0 = int(round((height / 2.0 - cy - h / 2.0) * pixels_per_um))
        y1 = int(round((height / 2.0 - cy + h / 2.0) * pixels_per_um))

        x0 = max(0, min(width_px - 1, x0))
        x1 = max(0, min(width_px, x1))
        y0 = max(0, min(height_px - 1, y0))
        y1 = max(0, min(height_px, y1))

        if x1 <= x0 or y1 <= y0:
            continue
        mask[y0:y1, x0:x1] = 1

    return mask


st.set_page_config(page_title="ELitho Simulation", layout="wide")

# CSS: comfortable margins
st.markdown(
    """
    <style>
    html, body, .main, .stApp {margin:0; padding:0}
    .block-container{max-width:100% !important; padding-left:1.25rem !important; padding-right:1.25rem !important}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("ELitho Simulation")


try:
    with open("defaults.json", "r", encoding="utf-8") as f:
        defaults = json.load(f)
except Exception:
    defaults = {
        "wavelength_nm": 13.5,
        "NA": 0.55,
        "incident_angle_deg": 0.0,
        "azimuthal_angle_deg": 0.0,
        "defocus_min_um": -1.0,
        "defocus_max_um": 1.0,
        "defocus_step_um": 0.1,
        "magnification_y": 8.0,
    }


def generate_intensity():
    pass


@st.cache_data
def generate_defocus_frames(
    defocus_list, wavelength_nm, NA, img_size=512, pixels_per_um=50
):
    frames = []
    profiles = []
    for d in defocus_list:
        sigma_factor = 1.0 + abs(float(d)) * 0.5
        img, r_um = simulate_psf(
            float(wavelength_nm),
            float(NA),
            img_size=img_size,
            pixels_per_um=pixels_per_um,
            sigma_factor=sigma_factor,
        )
        frames.append(img)
        center = img.shape[0] // 2
        profiles.append(img[center, :])
    return frames, profiles


def render_inputs():
    st.header("Optical Parameters")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        wavelength = st.number_input(
            "wavelength [nm]",
            min_value=0.0,
            value=float(defaults.get("wavelength_nm", 13.5)),
            step=0.1,
            format="%.1f",
            key="wavelength",
        )
        _na_options = [0.33, 0.55]
        _default_na = float(defaults.get("NA", 0.55))
        _default_index = (
            _na_options.index(_default_na) if _default_na in _na_options else 1
        )
        NA = st.selectbox(
            "NA",
            _na_options,
            index=_default_index,
            format_func=lambda x: f"{x:.2f}",
            key="NA",
        )
        prev_na = st.session_state.get("_prev_NA", None)
        if prev_na is None or float(prev_na) != float(NA):
            st.session_state["magnification_x"] = 4.0
            st.session_state["magnification_y"] = float(
                defaults.get("magnification_y", 8.0)
            )
            st.session_state["_prev_NA"] = float(NA)

    with c2:
        magnification_x = st.number_input(
            "X magnification",
            min_value=0.0,
            value=float(st.session_state.get("magnification_x", 4.0)),
            step=0.1,
            format="%.1f",
            key="magnification_x",
        )
        magnification_y = st.number_input(
            "Y magnification",
            min_value=0.0,
            value=float(
                st.session_state.get(
                    "magnification_y", defaults.get("magnification_y", 8.0)
                )
            ),
            step=0.1,
            format="%.1f",
            key="magnification_y",
        )
        if float(NA) == 0.55:
            central_obscuration = st.number_input(
                "Central obscuration [fraction]",
                min_value=0.0,
                max_value=1.0,
                value=0.0,
                step=0.01,
                format="%.2f",
                key="central_obscuration",
            )
        else:
            central_obscuration = 0.0

    with c3:
        incidence_angle = st.number_input(
            "incident angle [deg]",
            value=float(defaults.get("incidence_angle_deg", 0.0)),
            step=0.1,
            format="%.1f",
            key="incidence_angle",
        )
        azimuthal_angle = st.number_input(
            "azimuthal angle [deg]",
            value=float(defaults.get("azimuthal_angle_deg", 0.0)),
            step=0.1,
            format="%.1f",
            key="azimuthal_angle",
        )

    with c4:
        defocus_mode = st.radio(
            "Defocus mode",
            [
                "Single",
                "Sweep",
            ],
            horizontal=True,
        )
        if defocus_mode == "Single":
            defocus_min = st.number_input("Defocus value", value=0.0)
            defocus_max = None
            defocus_step = None
        else:
            defocus_min = st.number_input(
                "defocus min [μm]",
                value=float(defaults.get("defocus_min_um", -1.0)),
                step=0.1,
                format="%.3f",
                key="defocus_min",
            )
            defocus_max = st.number_input(
                "defocus max [μm]",
                value=float(defaults.get("defocus_max_um", 1.0)),
                step=0.1,
                format="%.3f",
                key="defocus_max",
            )
            defocus_step = st.number_input(
                "defocus step [μm]",
                min_value=0.0,
                value=float(defaults.get("defocus_step_um", 0.1)),
                step=0.01,
                format="%.3f",
                key="defocus_step",
            )

    st.markdown("---")

    st.header("Absorber Parameters")
    num_layers = st.number_input(
        "Number of layers", min_value=1, max_value=20, value=2, step=1, key="num_layers"
    )
    layers = []
    for li in range(1, int(num_layers) + 1):
        # Display the first layer as "Layer 1 (top)"; keep other layers unchanged.
        # Comments are written in English per project request.
        label = f"Layer {li} (top)" if li == 1 else f"Layer {li}"
        with st.expander(label, expanded=True):
            st.markdown("Complex refractive index")
            a1, a2 = st.columns(2)
            with a1:
                n_real = st.number_input(
                    "Real part (n)",
                    min_value=0.0,
                    value=1.0,
                    step=0.0001,
                    format="%.4f",
                    key=f"layer_{li}_n",
                )
            with a2:
                k_imag = st.number_input(
                    "Imag part (k)",
                    min_value=0.0,
                    value=0.0,
                    step=0.0001,
                    format="%.4f",
                    key=f"layer_{li}_k",
                )
            thickness = st.number_input(
                "Thickness [nm]",
                min_value=0.0,
                value=50.0,
                step=0.1,
                format="%.1f",
                key=f"layer_{li}_thickness",
            )
            layers.append(
                {
                    "thickness_nm": float(thickness),
                    "n": float(n_real),
                    "k": float(k_imag),
                }
            )

    st.markdown("---")

    st.header("Mask Pattern Parameters")
    mp_left, mp_right = st.columns([3, 1])
    with mp_left:
        tw, th = st.columns(2)
        with tw:
            mask_width = st.number_input(
                "Mask width [nm]",
                min_value=0.0,
                value=100.0,
                step=0.1,
                format="%.1f",
                key="mask_width",
            )
        with th:
            mask_height = st.number_input(
                "Mask height [nm]",
                min_value=0.0,
                value=100.0,
                step=0.1,
                format="%.1f",
                key="mask_height",
            )
        num_opens = st.number_input(
            "Number of opens",
            min_value=0,
            max_value=100,
            value=1,
            step=1,
            key="mask_num_opens",
        )
        opens = []
        for oi in range(1, int(num_opens) + 1):
            with st.expander(f"Open {oi}", expanded=True):
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    cx = st.number_input(
                        "center X [nm]",
                        value=0.0,
                        step=0.1,
                        format="%.1f",
                        key=f"mask_open_{oi}_cx",
                    )
                with c2:
                    cy = st.number_input(
                        "center Y [nm]",
                        value=0.0,
                        step=0.1,
                        format="%.1f",
                        key=f"mask_open_{oi}_cy",
                    )
                with c3:
                    ow = st.number_input(
                        "width [nm]",
                        min_value=0.0,
                        value=10.0,
                        step=0.1,
                        format="%.1f",
                        key=f"mask_open_{oi}_w",
                    )
                with c4:
                    oh = st.number_input(
                        "height [nm]",
                        min_value=0.0,
                        value=10.0,
                        step=0.1,
                        format="%.1f",
                        key=f"mask_open_{oi}_h",
                    )
                opens.append(
                    {
                        "center_x_um": float(cx),
                        "center_y_um": float(cy),
                        "width_um": float(ow),
                        "height_um": float(oh),
                    }
                )

    with mp_right:
        try:
            mask_arr = generate_mask(mask_width, mask_height, opens, pixels_per_um=5)
            fig_mask, axm = plt.subplots(figsize=(2, 2))
            axm.imshow(mask_arr, cmap="gray", origin="lower", interpolation="nearest")
            axm.axis("off")
            st.markdown("Mask preview (white = open)")
            st.pyplot(fig_mask)
        except Exception as e:
            st.error(f"Unable to render mask preview: {e}")

    st.markdown("---")

    st.header("Source Parameters")
    # Mesh input: allow the user to set the sampling mesh used in calculations.
    # Default to const.mesh when available, otherwise 0.50. Display with two decimals.
    mesh = st.number_input(
        "mesh",
        min_value=0.0,
        value=float(getattr(config, "mesh", 0.5)),
        step=0.01,
        format="%.2f",
        key="mesh",
    )

    sp_left, sp_right = st.columns([2, 2])
    with sp_left:
        # Source type selection: choose illumination type for the simulation.
        # Options correspond to elitho.const.IlluminationType members.
        _illum_options = [t.name for t in illumination.IlluminationType]
        _default_illum = _illum_options[0]
        _default_index = (
            _illum_options.index(_default_illum)
            if _default_illum in _illum_options
            else 0
        )
        source_type_name = st.selectbox(
            "Illumination type",
            _illum_options,
            index=_default_index,
            key="source_type",
        )
        # Map selected name to the enum value for downstream code compatibility.
        try:
            source_type_enum = illumination.IlluminationType[source_type_name]
        except Exception:
            source_type_enum = illumination.IlluminationType.CIRCULAR

        # Conditional inputs depending on the chosen source type.
        # initial parameters
        outer_sigma = 0.9
        inner_sigma = 0.55
        open_angle = 90.0

        if source_type_enum == illumination.IlluminationType.CIRCULAR:
            outer_sigma = st.number_input(
                "Outer sigma",
                min_value=0.0,
                value=outer_sigma,
                step=0.01,
                format="%.2f",
                key="outer_sigma",
            )
            ill = illumination.CircularIllumination(outer_sigma=float(outer_sigma))

        elif source_type_enum == illumination.IlluminationType.ANNULAR:
            outer_sigma = st.number_input(
                "Outer sigma",
                min_value=0.0,
                value=outer_sigma,
                step=0.01,
                format="%.2f",
                key="outer_sigma",
            )
            inner_sigma = st.number_input(
                "Inner sigma",
                min_value=0.0,
                value=inner_sigma,
                step=0.01,
                format="%.2f",
                key="inner_sigma",
            )
            ill = illumination.AnnularIllumination(
                outer_sigma=float(outer_sigma),
                inner_sigma=float(inner_sigma),
            )

        elif source_type_enum in (
            illumination.IlluminationType.DIPOLE_X,
            illumination.IlluminationType.DIPOLE_Y,
        ):
            outer_sigma = st.number_input(
                "Outer sigma",
                min_value=0.0,
                value=outer_sigma,
                step=0.01,
                format="%.2f",
                key="outer_sigma",
            )
            inner_sigma = st.number_input(
                "Inner sigma",
                min_value=0.0,
                value=inner_sigma,
                step=0.01,
                format="%.2f",
                key="inner_sigma",
            )
            open_angle = st.number_input(
                "Open angle [deg]",
                min_value=0.0,
                value=open_angle,
                step=0.1,
                format="%.1f",
                key="open_angle",
            )
            ill = illumination.DipoleIllumination(
                type=source_type_enum,
                outer_sigma=float(outer_sigma),
                inner_sigma=float(inner_sigma),
                open_angle=float(open_angle),
            )
        else:
            raise ValueError("Unsupported illumination type")

    sc = config.SimulationConfig(
        wavelength=float(wavelength),
        NA=float(NA),
        is_high_na=NA > 0.33,
        mask_width=float(mask_width),
        mask_height=float(mask_height),
        magnification_x=float(magnification_x),
        magnification_y=float(magnification_y),
        mesh=float(mesh),
        incidence_angle=float(incidence_angle),
        azimuthal_angle=float(azimuthal_angle),
        central_obscuration=float(central_obscuration),
        defocus_min=defocus_min,
        defocus_max=defocus_max,
        defocus_step=defocus_step,
    )

    with sp_right:
        try:
            k = 2.0 * np.pi / wavelength
            # dkx, dky, _ = source.uniform_k_source(sc, ill)
            dkx, dky, _, _, _ = source.get_valid_source_points(sc, ill)
            sxo = dkx / k / NA
            syo = dky / k / NA
            # Visualize source directions on a small figure using subplots
            ill_fig, ill_axi = plt.subplots()
            ill_axi.set_aspect("equal", adjustable="box")
            ill_axi.plot(sxo, syo, "o")
            ill_axi.set_xlim(-1.0, 1.0)
            ill_axi.set_ylim(-1.0, 1.0)
            st.markdown("**Illumination preview**")
            st.pyplot(ill_fig)
        except Exception as e:
            st.error(f"Unable to render illumination preview: {e}")

    return {
        "simulation_config": sc,
        "illumination": ill,
    }


# Layout: two columns (inputs / results)
main_col, side_col = st.columns([6, 4])
with main_col:
    params = render_inputs()

with side_col:
    st.header("Simulation Results")
    run = st.button("Run simulation")
    result_placeholder = st.empty()

    # When Run pressed: generate frames and persist into session_state
    if run:
        sc = params.get("simulation_config")
        ill = params.get("Illumination")

        with st.spinner("Generating frames for defocus sweep..."):
            frames, profiles = generate_defocus_frames(sc, ill)

        # # store generated frames under internal keys (avoid clashing with widget keys)
        # st.session_state["_generated_defocus_frames"] = frames
        # st.session_state["_generated_defocus_profiles"] = profiles
        # st.session_state["_generated_defocus_vals"] = defocus_vals
        # st.session_state["_generated_defocus_step"] = defocus_step_val
        # st.session_state["_generated_defocus_idx"] = 0

    # # If frames exist, show defocus-value slider and selected frame
    # if "_generated_defocus_frames" in st.session_state and st.session_state.get(
    #     "_generated_defocus_frames"
    # ):
    #     frames = st.session_state["_generated_defocus_frames"]
    #     profiles = st.session_state.get("_generated_defocus_profiles", [])
    #     defocus_vals = st.session_state.get("_generated_defocus_vals", [0.0])
    #     defocus_step_val = float(st.session_state.get("_generated_defocus_step", 0.1))

    #     if "_generated_defocus_idx" not in st.session_state:
    #         st.session_state["_generated_defocus_idx"] = 0

    #     try:
    #         def_min = float(defocus_vals[0])
    #         def_max = float(defocus_vals[-1])
    #     except Exception:
    #         def_min = 0.0
    #         def_max = 0.0

    #     default_idx = int(st.session_state.get("_generated_defocus_idx", 0))
    #     default_defocus = float(defocus_vals[default_idx]) if defocus_vals else 0.0

    #     defocus_selected = st.slider(
    #         "Defocus [nm]",
    #         min_value=def_min,
    #         max_value=def_max,
    #         value=default_defocus,
    #         step=float(defocus_step_val),
    #         format="%.3f",
    #         key="defocus_value_slider",
    #     )

    #     # map selected defocus value to nearest index
    #     try:
    #         idx = int(
    #             round(
    #                 (float(defocus_selected) - float(defocus_vals[0]))
    #                 / float(defocus_step_val)
    #             )
    #         )
    #     except Exception:
    #         idx = int(default_idx)
    #     idx = max(0, min(len(defocus_vals) - 1, int(idx)))
    #     st.session_state["_generated_defocus_idx"] = int(idx)

    #     sel_img = frames[int(idx)]
    #     sel_profile = profiles[int(idx)] if profiles else None

    #     fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    #     ax = axs[0]
    #     im = ax.imshow(sel_img, cmap="inferno", origin="lower")
    #     ax.set_title(f"PSF (defocus={defocus_vals[int(idx)]:.3f} μm)")
    #     ax.axis("off")
    #     fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    #     if sel_profile is not None:
    #         axs[1].plot(sel_profile)
    #         axs[1].set_title("Central cross-section")
    #         axs[1].set_xlabel("pixel")
    #         axs[1].set_ylabel("normalized intensity")
    #     else:
    #         axs[1].axis("off")

    #     fig.tight_layout()

    #     with result_placeholder.container():
    #         st.pyplot(fig)
