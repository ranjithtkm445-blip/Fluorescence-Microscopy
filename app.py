"""
app.py
─────────────────────────────────────────────────────────────────────────────
PURPOSE:
    Streamlit deployment app for Hugging Face Spaces.
    Images are preloaded from samples/ folder.
    User selects MoA class and site from dropdowns,
    clicks Run Prediction to see results and download PDF.

RUN:
    streamlit run app.py
─────────────────────────────────────────────────────────────────────────────
"""

import os
import io
import json
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import streamlit as st
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                 Table, TableStyle, Image as RLImage)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from step5_train import FusionModel, OMICS_DIM, NUM_CLASSES

# ── Paths ─────────────────────────────────────────────────────────────────────
SAMPLES_DIR    = "samples"
CHECKPOINT_DIR = "checkpoints"
IMAGE_SIZE     = 256

# ── Class names and colors ────────────────────────────────────────────────────
CLASS_NAMES = [
    'Actin disruptors',
    'Aurora kinase inhibitors',
    'Microtubule destabilizers',
    'Microtubule stabilizers',
]
CLASS_COLORS = {
    'Actin disruptors'         : '#2196F3',
    'Aurora kinase inhibitors' : '#4CAF50',
    'Microtubule destabilizers': '#FF9800',
    'Microtubule stabilizers'  : '#E91E63',
}


# ══════════════════════════════════════════════════════════════════════════════
# Load model — cached so it only loads once
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_model():
    device    = torch.device('cpu')
    model     = FusionModel(num_classes=NUM_CLASSES).to(device)
    ckpt_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")
    model.load_state_dict(torch.load(ckpt_path,
                                     map_location=device,
                                     weights_only=True))
    model.eval()
    return model, device


# ══════════════════════════════════════════════════════════════════════════════
# Load sample index — cached so it only loads once
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data
def load_index():
    index_path = os.path.join(SAMPLES_DIR, "index.json")
    with open(index_path, 'r') as f:
        return json.load(f)


# ══════════════════════════════════════════════════════════════════════════════
# Load PNG sample images for one site
# ══════════════════════════════════════════════════════════════════════════════

def load_site_images(site_idx):
    """
    Load pre-exported PNG channel images for one site.
    Returns dict of normalized numpy arrays (H, W) in [0, 1].
    """
    channels = {}
    prefix   = f"site_{site_idx:03d}"

    for ch_name in ['DAPI', 'Tubulin', 'Actin']:
        path = os.path.join(SAMPLES_DIR, f"{prefix}_{ch_name}.png")
        if os.path.exists(path):
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            img = img.astype(np.float32) / 255.0
            img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE),
                             interpolation=cv2.INTER_AREA)
            channels[ch_name] = img
        else:
            channels[ch_name] = np.zeros((IMAGE_SIZE, IMAGE_SIZE),
                                          dtype=np.float32)
    return channels


# ══════════════════════════════════════════════════════════════════════════════
# Build image tensor
# ══════════════════════════════════════════════════════════════════════════════

def build_image_tensor(channels, device):
    img = np.stack([
        channels['DAPI'],
        channels['Tubulin'],
        channels['Actin']
    ], axis=0).astype(np.float32)
    return torch.tensor(img).unsqueeze(0).to(device)


# ══════════════════════════════════════════════════════════════════════════════
# Synthetic omics vector
# ══════════════════════════════════════════════════════════════════════════════

def get_omics_vector(compound, device):
    seed = abs(hash(compound)) % (2**31)
    rng  = np.random.RandomState(seed)
    vec  = rng.randn(OMICS_DIM).astype(np.float32)
    return torch.tensor(vec).unsqueeze(0).to(device)


# ══════════════════════════════════════════════════════════════════════════════
# GradCAM
# PURPOSE: Highlight which image regions drove the model prediction.
# ══════════════════════════════════════════════════════════════════════════════

class GradCAM:
    def __init__(self, model):
        self.model       = model
        self.gradients   = None
        self.activations = None
        target_layer     = model.image_branch.resnet.layer4[-1]
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def compute(self, image_tensor, omics_tensor, class_idx=None):
        self.model.zero_grad()
        output = self.model(image_tensor, omics_tensor)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        score = output[0, class_idx]
        score.backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam     = (weights * self.activations).sum(dim=1, keepdim=True)
        cam     = F.relu(cam)
        cam     = F.interpolate(cam, size=(IMAGE_SIZE, IMAGE_SIZE),
                                mode='bilinear', align_corners=False)
        cam     = cam.squeeze().cpu().numpy()
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        return cam, class_idx


# ══════════════════════════════════════════════════════════════════════════════
# GradCAM overlay
# ══════════════════════════════════════════════════════════════════════════════

def overlay_gradcam(channel_img, cam):
    base     = (channel_img * 255).astype(np.uint8)
    base_rgb = cv2.cvtColor(base, cv2.COLOR_GRAY2RGB)
    heatmap  = (cam * 255).astype(np.uint8)
    heatmap  = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap  = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay  = cv2.addWeighted(base_rgb, 0.5, heatmap, 0.5, 0)
    return overlay


# ══════════════════════════════════════════════════════════════════════════════
# Save matplotlib figure to bytes
# ══════════════════════════════════════════════════════════════════════════════

def fig_to_bytes(fig, dpi=150):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
    buf.seek(0)
    return buf.read()


# ══════════════════════════════════════════════════════════════════════════════
# Generate PDF report
# ══════════════════════════════════════════════════════════════════════════════

def generate_pdf(site_meta, channels, cam, probs, pred_class, pred_conf):
    buf    = io.BytesIO()
    doc    = SimpleDocTemplate(buf, pagesize=A4,
                               topMargin=1.5*cm, bottomMargin=1.5*cm,
                               leftMargin=1.5*cm, rightMargin=1.5*cm)
    styles = getSampleStyleSheet()
    story  = []

    # Title
    story.append(Paragraph(
        "BBBC021 MoA Prediction Report",
        ParagraphStyle('Title', parent=styles['Title'],
                       fontSize=18, spaceAfter=6,
                       alignment=TA_CENTER)
    ))
    story.append(Paragraph(
        "Multi-Modal Fluorescence Microscopy Analysis",
        ParagraphStyle('Sub', parent=styles['Normal'],
                       fontSize=11, alignment=TA_CENTER,
                       textColor=colors.grey, spaceAfter=16)
    ))
    story.append(Spacer(1, 0.3*cm))

    # Site information
    story.append(Paragraph("Site Information", styles['Heading2']))
    info_data = [
        ['Field',         'Value'],
        ['Plate',         site_meta['plate']],
        ['Compound',      site_meta['compound']],
        ['Concentration', f"{site_meta['concentration']} μM"],
        ['True MoA',      site_meta['moa']],
    ]
    info_table = Table(info_data, colWidths=[5*cm, 10*cm])
    info_table.setStyle(TableStyle([
        ('BACKGROUND',     (0,0), (-1,0), colors.HexColor('#1976D2')),
        ('TEXTCOLOR',      (0,0), (-1,0), colors.white),
        ('FONTNAME',       (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE',       (0,0), (-1,-1), 10),
        ('ROWBACKGROUNDS', (0,1), (-1,-1),
         [colors.HexColor('#F5F5F5'), colors.white]),
        ('GRID',           (0,0), (-1,-1), 0.5, colors.grey),
        ('PADDING',        (0,0), (-1,-1), 6),
    ]))
    story.append(info_table)
    story.append(Spacer(1, 0.5*cm))

    # Prediction result
    story.append(Paragraph("Prediction Result", styles['Heading2']))
    correct      = pred_class == site_meta['moa']
    result_color = (colors.HexColor('#4CAF50')
                    if correct else colors.HexColor('#F44336'))
    pred_data    = [
        ['Predicted MoA', 'Confidence', 'Result'],
        [pred_class,
         f"{pred_conf*100:.1f}%",
         "Correct" if correct else "Incorrect"],
    ]
    pred_table = Table(pred_data, colWidths=[7*cm, 4*cm, 4*cm])
    pred_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1976D2')),
        ('TEXTCOLOR',  (0,0), (-1,0), colors.white),
        ('FONTNAME',   (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTNAME',   (0,1), (-1,1), 'Helvetica-Bold'),
        ('FONTSIZE',   (0,0), (-1,-1), 11),
        ('BACKGROUND', (2,1), (2,1),  result_color),
        ('TEXTCOLOR',  (2,1), (2,1),  colors.white),
        ('GRID',       (0,0), (-1,-1), 0.5, colors.grey),
        ('ALIGN',      (0,0), (-1,-1), 'CENTER'),
        ('PADDING',    (0,0), (-1,-1), 8),
    ]))
    story.append(pred_table)
    story.append(Spacer(1, 0.5*cm))

    # Confidence scores
    story.append(Paragraph("Confidence Scores", styles['Heading2']))
    conf_data = [['MoA Class', 'Confidence']]
    for cls, prob in zip(CLASS_NAMES, probs):
        conf_data.append([cls, f"{prob*100:.1f}%"])
    conf_table = Table(conf_data, colWidths=[10*cm, 5*cm])
    conf_table.setStyle(TableStyle([
        ('BACKGROUND',     (0,0), (-1,0), colors.HexColor('#1976D2')),
        ('TEXTCOLOR',      (0,0), (-1,0), colors.white),
        ('FONTNAME',       (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE',       (0,0), (-1,-1), 10),
        ('ROWBACKGROUNDS', (0,1), (-1,-1),
         [colors.HexColor('#F5F5F5'), colors.white]),
        ('GRID',           (0,0), (-1,-1), 0.5, colors.grey),
        ('PADDING',        (0,0), (-1,-1), 6),
    ]))
    story.append(conf_table)
    story.append(Spacer(1, 0.5*cm))

    # Channel images
    story.append(Paragraph("Fluorescence Channels", styles['Heading2']))
    cmaps   = {'DAPI': 'Blues', 'Tubulin': 'Greens', 'Actin': 'Reds'}
    ch_imgs = []
    for ch_name in ['DAPI', 'Tubulin', 'Actin']:
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.imshow(channels[ch_name], cmap=cmaps[ch_name],
                  vmin=0, vmax=1)
        ax.set_title(ch_name, fontsize=10)
        ax.axis('off')
        plt.tight_layout()
        img_bytes = fig_to_bytes(fig)
        plt.close()
        ch_imgs.append(RLImage(io.BytesIO(img_bytes),
                               width=5.5*cm, height=5.5*cm))
    story.append(Table([ch_imgs], colWidths=[6*cm, 6*cm, 6*cm]))
    story.append(Spacer(1, 0.5*cm))

    # GradCAM overlays
    story.append(Paragraph(
        "GradCAM — Regions Driving Prediction",
        styles['Heading2']
    ))
    story.append(Paragraph(
        "Warmer colors (red/yellow) = regions the model focused on. "
        "Cooler colors (blue) = less important regions.",
        ParagraphStyle('Caption', parent=styles['Normal'],
                       fontSize=9, textColor=colors.grey,
                       spaceAfter=8)
    ))
    grad_imgs = []
    for ch_name in ['DAPI', 'Tubulin', 'Actin']:
        overlay = overlay_gradcam(channels[ch_name], cam)
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.imshow(overlay)
        ax.set_title(f"GradCAM — {ch_name}", fontsize=10)
        ax.axis('off')
        plt.tight_layout()
        img_bytes = fig_to_bytes(fig)
        plt.close()
        grad_imgs.append(RLImage(io.BytesIO(img_bytes),
                                  width=5.5*cm, height=5.5*cm))
    story.append(Table([grad_imgs], colWidths=[6*cm, 6*cm, 6*cm]))
    story.append(Spacer(1, 0.5*cm))

    # Footer
    story.append(Paragraph(
        "Generated by BBBC021 MoA Predictor — "
        "Multi-Modal Fluorescence Microscopy Analysis Pipeline",
        ParagraphStyle('Footer', parent=styles['Normal'],
                       fontSize=8, textColor=colors.grey,
                       alignment=TA_CENTER)
    ))

    doc.build(story)
    buf.seek(0)
    return buf.read()


# ══════════════════════════════════════════════════════════════════════════════
# Main Streamlit App
# ══════════════════════════════════════════════════════════════════════════════

def main():
    st.set_page_config(
        page_title = "BBBC021 MoA Predictor",
        page_icon  = "🔬",
        layout     = "wide"
    )

    st.title("🔬 Drug Mechanism of Action Predictor")
    st.markdown(
        "Multi-modal fluorescence microscopy analysis using "
        "ResNet18 image branch + synthetic omics fusion. "
        "Select a MoA class and site, then click **Run Prediction**."
    )

    # ── Load model and index ──────────────────────────────────────────────
    with st.spinner("Loading model..."):
        model, device = load_model()
        index         = load_index()
        gradcam       = GradCAM(model)

    # ── Sidebar ───────────────────────────────────────────────────────────
    st.sidebar.header("Select imaging site")

    # MoA class selector
    moa_classes  = sorted(set(s['moa'] for s in index))
    selected_moa = st.sidebar.selectbox("MoA class", moa_classes)

    # Site selector filtered by MoA class
    class_sites  = [s for s in index if s['moa'] == selected_moa]
    site_labels  = [
        f"Site {s['site_idx']} — "
        f"{s['compound'].title()} "
        f"({s['concentration']} μM)"
        for s in class_sites
    ]
    selected_label = st.sidebar.selectbox("Site", site_labels)
    selected_site  = class_sites[site_labels.index(selected_label)]

    # Site info
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Plate:** {selected_site['plate']}")
    st.sidebar.markdown(
        f"**Compound:** {selected_site['compound'].title()}"
    )
    st.sidebar.markdown(
        f"**Concentration:** {selected_site['concentration']} μM"
    )
    st.sidebar.markdown(f"**True MoA:** {selected_site['moa']}")

    st.markdown("---")

    # ── Run Prediction button ─────────────────────────────────────────────
    run_clicked = st.button(
        "▶ Run Prediction",
        type                = "primary",
        use_container_width = True
    )

    if run_clicked:

        site_idx = selected_site['site_idx']
        compound = selected_site['compound']
        true_moa = selected_site['moa']

        # ── Load images ───────────────────────────────────────────────
        with st.spinner("Loading images..."):
            channels = load_site_images(site_idx)

        # ── Show channel images ───────────────────────────────────────
        st.markdown("### Fluorescence channels")
        cmaps      = {'DAPI': 'Blues', 'Tubulin': 'Greens', 'Actin': 'Reds'}
        c1, c2, c3 = st.columns(3)
        for col, ch_name in zip([c1, c2, c3],
                                 ['DAPI', 'Tubulin', 'Actin']):
            with col:
                fig, ax = plt.subplots(figsize=(4, 4))
                ax.imshow(channels[ch_name],
                          cmap=cmaps[ch_name], vmin=0, vmax=1)
                ax.set_title(ch_name, fontsize=11)
                ax.axis('off')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

        # ── Run model ─────────────────────────────────────────────────
        with st.spinner("Running prediction and GradCAM..."):
            image_tensor = build_image_tensor(channels, device)
            omics_tensor = get_omics_vector(compound, device)

            with torch.no_grad():
                output = model(image_tensor, omics_tensor)
                probs  = torch.softmax(output, dim=1)\
                              .squeeze().cpu().numpy()

            pred_idx   = int(probs.argmax())
            pred_class = CLASS_NAMES[pred_idx]
            pred_conf  = float(probs[pred_idx])

            # GradCAM
            img_tensor_grad = build_image_tensor(channels, device)
            img_tensor_grad.requires_grad_(True)
            cam, _ = gradcam.compute(
                img_tensor_grad, omics_tensor, class_idx=pred_idx
            )

        # ── Prediction result ─────────────────────────────────────────
        st.markdown("### Prediction result")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Predicted MoA", pred_class)
        with col2:
            st.metric("Confidence", f"{pred_conf*100:.1f}%")
        with col3:
            correct = pred_class == true_moa
            st.metric(
                "True MoA", true_moa,
                delta      = "✓ Correct" if correct else "✗ Incorrect",
                delta_color= "normal" if correct else "inverse"
            )

        # ── Confidence bar chart ──────────────────────────────────────
        st.markdown("### Confidence scores")
        fig, ax    = plt.subplots(figsize=(8, 2.5))
        bar_colors = [CLASS_COLORS.get(c, '#888888') for c in CLASS_NAMES]
        bars       = ax.barh(CLASS_NAMES, probs * 100, color=bar_colors)
        ax.set_xlabel("Confidence (%)")
        ax.set_xlim(0, 100)
        for bar, prob in zip(bars, probs):
            ax.text(bar.get_width() + 1,
                    bar.get_y() + bar.get_height()/2,
                    f"{prob*100:.1f}%", va='center', fontsize=9)
        ax.axvline(x=50, color='gray', linestyle='--', linewidth=0.8)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # ── GradCAM overlays ──────────────────────────────────────────
        st.markdown("### GradCAM — regions driving the prediction")
        st.caption(
            "Warmer colors (red/yellow) = regions the model focused on. "
            "Cooler colors (blue) = less important regions."
        )
        g1, g2, g3 = st.columns(3)
        for col, ch_name in zip([g1, g2, g3],
                                 ['DAPI', 'Tubulin', 'Actin']):
            with col:
                overlay = overlay_gradcam(channels[ch_name], cam)
                fig, ax = plt.subplots(figsize=(4, 4))
                ax.imshow(overlay)
                ax.set_title(f"GradCAM — {ch_name}", fontsize=11)
                ax.axis('off')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

        # ── PDF download ──────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### Download report")
        with st.spinner("Generating PDF..."):
            pdf_bytes = generate_pdf(
                selected_site, channels, cam,
                probs, pred_class, pred_conf
            )

        st.download_button(
            label               = "📄 Download PDF Report",
            data                = pdf_bytes,
            file_name           = f"moa_report_{compound}_{site_idx}.pdf",
            mime                = "application/pdf",
            use_container_width = True
        )


if __name__ == "__main__":
    main()