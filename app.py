"""
Image to DXF Converter - Web Application
A beautifully designed web service for converting bitmap images to DXF format
"""

import streamlit as st
from PIL import Image, ImageDraw, ImageFilter, ImageOps
import io
import base64
import math
from converter import BitmapToDXFConverter

# Page configuration
st.set_page_config(
    page_title="Image to DXF Converter",
    page_icon="☕",
    layout="wide"
)

# Beautiful custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700&family=Source+Sans+Pro:wght@300;400;600&display=swap');
    
    /* Global styles */
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Main container */
    .main .block-container {
        padding-top: 2rem;
        max-width: 1400px;
    }
    
    /* Typography */
    h1, h2, h3 {
        font-family: 'Playfair Display', Georgia, serif !important;
        color: #f8f0e3 !important;
    }
    
    p, li, span, label {
        font-family: 'Source Sans Pro', sans-serif !important;
    }
    
    /* Hero header */
    .hero-container {
        text-align: center;
        padding: 2rem 1rem;
        margin-bottom: 2rem;
        background: linear-gradient(180deg, rgba(232,180,120,0.1) 0%, rgba(0,0,0,0) 100%);
        border-radius: 20px;
    }
    
    .hero-title {
        font-family: 'Playfair Display', Georgia, serif;
        font-size: 3.5rem;
        font-weight: 700;
        color: #e8b478;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        letter-spacing: 2px;
    }
    
    .hero-subtitle {
        font-family: 'Source Sans Pro', sans-serif;
        font-size: 1.2rem;
        color: #a0a0b0;
        font-weight: 300;
        letter-spacing: 1px;
    }
    
    /* Cards */
    .custom-card {
        background: rgba(255,255,255,0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(232,180,120,0.2);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    
    .custom-card:hover {
        border-color: rgba(232,180,120,0.4);
        box-shadow: 0 8px 32px rgba(232,180,120,0.1);
    }
    
    .card-title {
        font-family: 'Playfair Display', Georgia, serif;
        font-size: 1.4rem;
        color: #e8b478;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Coffee tier cards */
    .coffee-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 1rem;
        margin: 1rem 0;
    }
    
    .coffee-card {
        background: linear-gradient(145deg, rgba(255,255,255,0.08) 0%, rgba(255,255,255,0.02) 100%);
        border: 1px solid rgba(232,180,120,0.3);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .coffee-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #e8b478, #d4a574, #c9956c);
    }
    
    .coffee-card:hover {
        transform: translateY(-4px);
        border-color: #e8b478;
        box-shadow: 0 12px 40px rgba(232,180,120,0.2);
    }
    
    .coffee-icon {
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    
    .coffee-name {
        font-family: 'Playfair Display', Georgia, serif;
        font-size: 1.3rem;
        color: #f8f0e3;
        margin-bottom: 0.3rem;
    }
    
    .coffee-price {
        font-family: 'Source Sans Pro', sans-serif;
        font-size: 2rem;
        font-weight: 600;
        color: #e8b478;
        margin: 0.5rem 0;
    }
    
    .coffee-desc {
        font-family: 'Source Sans Pro', sans-serif;
        font-size: 0.9rem;
        color: #a0a0b0;
        line-height: 1.4;
    }
    
    /* Preview container */
    .preview-container {
        background: rgba(0,0,0,0.3);
        border: 2px solid rgba(232,180,120,0.3);
        border-radius: 12px;
        padding: 1rem;
        min-height: 300px;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    
    .preview-label {
        font-family: 'Source Sans Pro', sans-serif;
        font-size: 0.85rem;
        color: #e8b478;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 0.5rem;
    }
    
    /* Stats display */
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 1rem;
        margin: 1rem 0;
    }
    
    .stat-box {
        background: rgba(232,180,120,0.1);
        border: 1px solid rgba(232,180,120,0.3);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
    }
    
    .stat-value {
        font-family: 'Playfair Display', Georgia, serif;
        font-size: 1.8rem;
        color: #e8b478;
        font-weight: 600;
    }
    
    .stat-label {
        font-family: 'Source Sans Pro', sans-serif;
        font-size: 0.85rem;
        color: #a0a0b0;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Buttons */
    .stButton > button {
        font-family: 'Source Sans Pro', sans-serif !important;
        font-weight: 600 !important;
        letter-spacing: 1px !important;
        border-radius: 8px !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 20px rgba(232,180,120,0.3) !important;
    }
    
    /* Download button special styling */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #e8b478 0%, #d4a574 100%) !important;
        color: #1a1a2e !important;
        font-weight: 600 !important;
        border: none !important;
        padding: 0.75rem 2rem !important;
    }
    
    /* File uploader */
    .stFileUploader > div {
        background: rgba(255,255,255,0.05) !important;
        border: 2px dashed rgba(232,180,120,0.4) !important;
        border-radius: 12px !important;
    }
    
    /* Sliders */
    .stSlider > div > div {
        background: rgba(232,180,120,0.3) !important;
    }
    
    /* Select boxes */
    .stSelectbox > div > div {
        background: rgba(255,255,255,0.05) !important;
        border-color: rgba(232,180,120,0.3) !important;
    }
    
    /* Info boxes */
    .stAlert {
        background: rgba(232,180,120,0.1) !important;
        border: 1px solid rgba(232,180,120,0.3) !important;
        border-radius: 12px !important;
    }
    
    /* Divider */
    hr {
        border-color: rgba(232,180,120,0.2) !important;
        margin: 2rem 0 !important;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        margin-top: 3rem;
        border-top: 1px solid rgba(232,180,120,0.2);
    }
    
    .footer-text {
        font-family: 'Source Sans Pro', sans-serif;
        color: #606070;
        font-size: 0.9rem;
    }
    
    /* Responsive adjustments */
    @media (max-width: 768px) {
        .hero-title {
            font-size: 2.5rem;
        }
        .coffee-grid {
            grid-template-columns: 1fr;
        }
        .stats-grid {
            grid-template-columns: 1fr;
        }
    }
</style>
""", unsafe_allow_html=True)

# Payment links (Stripe - LIVE)
PAYMENT_LINKS = {
    "deli_coffee": "https://buy.stripe.com/4gM7sK7524Lrarof3Ibo400",
    "cappuccino": "https://buy.stripe.com/5kQ6oG896b9PdDAcVAbo401",
    "specialty_espresso": "https://buy.stripe.com/dRm28qcpmgu9fLI6xcbo402"
}

# Initialize converter
converter = BitmapToDXFConverter()


def generate_preview(image, mode, threshold, invert, line_step):
    """Generate a visual preview of what the DXF output will look like."""
    # Work with a smaller version for preview
    max_preview_size = 400
    img = image.copy().convert('L')
    
    # Resize for preview
    ratio = min(max_preview_size / img.size[0], max_preview_size / img.size[1])
    new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
    img = img.resize(new_size, Image.Resampling.LANCZOS)
    
    w, h = img.size
    
    # Scale line_step proportionally
    preview_line_step = max(1, int(line_step * ratio))
    
    # Create preview image (white background, black lines)
    preview = Image.new('RGB', (w, h), '#1a1a2e')
    draw = ImageDraw.Draw(preview)
    
    px = img.load()
    
    if mode == "floyd_steinberg":
        # Apply dithering
        if invert:
            img = ImageOps.invert(img)
        
        pixels = list(img.getdata())
        data = [[float(pixels[y * w + x]) for x in range(w)] for y in range(h)]
        
        for y in range(h):
            for x in range(w):
                old_pixel = data[y][x]
                new_pixel = 255 if old_pixel > 127 else 0
                data[y][x] = new_pixel
                error = old_pixel - new_pixel
                
                if x + 1 < w:
                    data[y][x + 1] += error * 7 / 16
                if y + 1 < h:
                    if x > 0:
                        data[y + 1][x - 1] += error * 3 / 16
                    data[y + 1][x] += error * 5 / 16
                    if x + 1 < w:
                        data[y + 1][x + 1] += error * 1 / 16
        
        # Draw dots
        for y in range(0, h, preview_line_step):
            for x in range(0, w, preview_line_step):
                if data[y][x] < 128:
                    draw.ellipse([x-1, y-1, x+1, y+1], fill='#e8b478')
    
    elif mode == "outline":
        # Simple edge detection for preview
        if invert:
            img = ImageOps.invert(img)
        
        edges = img.filter(ImageFilter.FIND_EDGES)
        edges_px = edges.load()
        
        for y in range(h):
            for x in range(w):
                if edges_px[x, y] > 50:
                    draw.point((x, y), fill='#e8b478')
    
    else:  # threshold mode
        def is_black(v):
            vb = v < threshold
            return (not vb) if invert else vb
        
        # Draw horizontal line segments
        for y in range(0, h, preview_line_step):
            x = 0
            while x < w:
                # Find start of black region
                while x < w and not is_black(px[x, y]):
                    x += 1
                if x >= w:
                    break
                x1 = x
                # Find end of black region
                while x < w and is_black(px[x, y]):
                    x += 1
                x2 = x
                # Draw line
                draw.line([(x1, y), (x2, y)], fill='#e8b478', width=1)
    
    return preview


def image_to_base64(img):
    """Convert PIL Image to base64 string for display."""
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


# ============== MAIN APP ==============

# Hero header
st.markdown("""
<div class="hero-container">
    <div class="hero-title">☕ Image to DXF</div>
    <div class="hero-subtitle">Transform your images into precision laser-ready DXF files</div>
</div>
""", unsafe_allow_html=True)

# Main layout
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    # Upload section
    st.markdown("""
    <div class="card-title">📁 Upload Your Image</div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Drag and drop or click to browse",
        type=["png", "jpg", "jpeg", "bmp", "gif", "tiff"],
        label_visibility="collapsed"
    )
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption=f"Original: {image.size[0]}×{image.size[1]} px", use_container_width=True)
        
        # Settings section
        st.markdown("""
        <div class="card-title" style="margin-top: 2rem;">⚙️ Conversion Settings</div>
        """, unsafe_allow_html=True)
        
        mode = st.selectbox(
            "Conversion Mode",
            options=["threshold", "floyd_steinberg", "outline"],
            format_func=lambda x: {
                "threshold": "🔲 Threshold (Lines)",
                "floyd_steinberg": "⚫ Floyd-Steinberg (Dots)",
                "outline": "✏️ Outline (Contours)"
            }[x]
        )
        
        col_s1, col_s2 = st.columns(2)
        
        with col_s1:
            image_height = st.number_input(
                "Output Height (µm)",
                min_value=100.0,
                max_value=100000.0,
                value=1000.0,
                step=100.0
            )
        
        with col_s2:
            spot_size = st.number_input(
                "Spot Size (µm)",
                min_value=1.0,
                max_value=100.0,
                value=5.0,
                step=1.0
            )
        
        if mode == "threshold":
            threshold = st.slider("Threshold", 0, 255, 200)
        else:
            threshold = 200
        
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            invert = st.checkbox("Invert colors", value=False)
            flip_y = st.checkbox("Flip Y axis", value=True)
        with col_c2:
            bidirectional = st.checkbox("Bidirectional scan", value=True)
            if mode == "outline":
                outline_levels = st.slider("Contour levels", 2, 16, 2)
                smoothing = st.slider("Smoothing", 0.0, 10.0, 2.0)
            else:
                outline_levels = 2
                smoothing = 2.0

with col_right:
    if uploaded_file:
        # Preview section
        st.markdown("""
        <div class="card-title">👁️ Live Preview</div>
        <div class="preview-label">How your DXF will look</div>
        """, unsafe_allow_html=True)
        
        # Calculate line step for preview
        w, h = image.size
        aspect_ratio = w / h
        image_width_um = image_height * aspect_ratio
        pixel_size_y = image_height / h
        min_spacing = spot_size * 1.1
        line_step = max(1, int(math.ceil(min_spacing / pixel_size_y)))
        
        # Generate and display preview
        preview_img = generate_preview(image, mode, threshold, invert, line_step)
        st.image(preview_img, use_container_width=True)
        
        # Stats
        st.markdown("""
        <div class="stats-grid">
            <div class="stat-box">
                <div class="stat-value">{:.0f}</div>
                <div class="stat-label">Width (µm)</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">{:.0f}</div>
                <div class="stat-label">Height (µm)</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">{}</div>
                <div class="stat-label">Line Step</div>
            </div>
        </div>
        """.format(image_width_um, image_height, line_step), unsafe_allow_html=True)
    else:
        # Placeholder
        st.markdown("""
        <div class="custom-card" style="min-height: 400px; display: flex; align-items: center; justify-content: center;">
            <div style="text-align: center; color: #606070;">
                <div style="font-size: 4rem; margin-bottom: 1rem;">🖼️</div>
                <div style="font-family: 'Source Sans Pro', sans-serif;">
                    Upload an image to see the preview
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# Divider
st.markdown("<hr>", unsafe_allow_html=True)

# Payment section
if uploaded_file:
    st.markdown("""
    <div style="text-align: center; margin-bottom: 1.5rem;">
        <div class="card-title" style="justify-content: center;">☕ Support the Developer</div>
        <p style="color: #a0a0b0; font-family: 'Source Sans Pro', sans-serif;">
            Choose a coffee based on your image complexity
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="coffee-grid">
        <div class="coffee-card">
            <div class="coffee-icon">🥤</div>
            <div class="coffee-name">Deli Coffee</div>
            <div class="coffee-price">$2.50</div>
            <div class="coffee-desc">Perfect for simple graphics<br>and basic shapes</div>
        </div>
        <div class="coffee-card">
            <div class="coffee-icon">☕</div>
            <div class="coffee-name">Cappuccino</div>
            <div class="coffee-price">$5.00</div>
            <div class="coffee-desc">For average complexity<br>and detailed images</div>
        </div>
        <div class="coffee-card">
            <div class="coffee-icon">✨</div>
            <div class="coffee-name">Specialty Espresso</div>
            <div class="coffee-price">$6.50</div>
            <div class="coffee-desc">For complex files<br>and high-detail work</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        st.link_button("Buy Deli Coffee", PAYMENT_LINKS["deli_coffee"], use_container_width=True)
    with col_p2:
        st.link_button("Buy Cappuccino", PAYMENT_LINKS["cappuccino"], use_container_width=True)
    with col_p3:
        st.link_button("Buy Specialty", PAYMENT_LINKS["specialty_espresso"], use_container_width=True)
    
    st.info("💡 Payment is on the honor system. Support the project, then convert your file below!")
    
    # Convert section
    st.markdown("<hr>", unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center;">
        <div class="card-title" style="justify-content: center;">🚀 Convert & Download</div>
    </div>
    """, unsafe_allow_html=True)
    
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        convert_clicked = st.button("⚡ Convert to DXF", type="primary", use_container_width=True)
    
    if convert_clicked:
        with st.spinner("Converting your image..."):
            try:
                uploaded_file.seek(0)
                image_bytes = uploaded_file.read()
                
                dxf_content, stats = converter.convert(
                    input_image=image_bytes,
                    mode=mode,
                    image_height_um=image_height,
                    spot_size_um=spot_size,
                    spot_spacing_factor=1.1,
                    threshold=threshold,
                    invert=invert,
                    flip_y=flip_y,
                    bidirectional=bidirectional,
                    outline_levels=outline_levels,
                    smoothing_amount=smoothing,
                    corner_threshold=45.0
                )
                
                st.session_state['dxf_content'] = dxf_content
                st.session_state['stats'] = stats
                st.session_state['filename'] = uploaded_file.name.rsplit('.', 1)[0] + '.dxf'
                
                st.success("✅ Conversion complete!")
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    
    # Download section
    if 'dxf_content' in st.session_state:
        stats = st.session_state['stats']
        
        st.markdown("""
        <div class="stats-grid">
            <div class="stat-box">
                <div class="stat-value">{:,}</div>
                <div class="stat-label">Entities</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">{:.0f}×{:.0f}</div>
                <div class="stat-label">Size (µm)</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">{:.1f} KB</div>
                <div class="stat-label">File Size</div>
            </div>
        </div>
        """.format(
            stats['entity_count'],
            stats['width_um'],
            stats['height_um'],
            stats['file_size_kb']
        ), unsafe_allow_html=True)
        
        col_d1, col_d2, col_d3 = st.columns([1, 2, 1])
        with col_d2:
            st.download_button(
                label="⬇️ Download DXF File",
                data=st.session_state['dxf_content'],
                file_name=st.session_state['filename'],
                mime="application/dxf",
                use_container_width=True
            )

# Footer
st.markdown("""
<div class="footer">
    <div class="footer-text">
        Image to DXF Converter • Built with ❤️ for the maker community<br>
        <span style="font-size: 0.8rem;">Precision tools for laser cutting & CNC</span>
    </div>
</div>
""", unsafe_allow_html=True)
