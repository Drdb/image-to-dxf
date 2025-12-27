PIL.UnidentifiedImageError: This app has encountered an error. The original error message is redacted to prevent data leaks. Full error details have been recorded in the logs (if you're on Streamlit Cloud, click on 'Manage app' in the lower right of your app).
Traceback:
File "/mount/src/image-to-dxf/app.py", line 608, in <module>
    st.image("examples/simple_outline.png", use_container_width=True)
    ~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "/home/adminuser/venv/lib/python3.13/site-packages/streamlit/runtime/metrics_util.py", line 531, in wrapped_func
    result = non_optional_func(*args, **kwargs)
File "/home/adminuser/venv/lib/python3.13/site-packages/streamlit/elements/image.py", line 206, in image
    marshall_images(
    ~~~~~~~~~~~~~~~^
        self.dg._get_delta_path_str(),
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<6 lines>...
        output_format,
        ^^^^^^^^^^^^^^
    )
    ^
File "/home/adminuser/venv/lib/python3.13/site-packages/streamlit/elements/lib/image_utils.py", line 445, in marshall_images
    proto_img.url = image_to_url(
                    ~~~~~~~~~~~~^
        single_image, layout_config, clamp, channels, output_format, image_id
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
File "/home/adminuser/venv/lib/python3.13/site-packages/streamlit/elements/lib/image_utils.py", line 336, in image_to_url
    image_format = _validate_image_format_string(image_data, output_format)
File "/home/adminuser/venv/lib/python3.13/site-packages/streamlit/elements/lib/image_utils.py", line 111, in _validate_image_format_string
    pil_image = Image.open(io.BytesIO(image_data))
File "/home/adminuser/venv/lib/python3.13/site-packages/PIL/Image.py", line 3560, in open
    raise UnidentifiedImageError(msg)
