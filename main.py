import gradio as gr
import numpy as np
import tensorflow as tf
import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
model=tf.keras.models.load_model('digit_recognition.keras')


def sketchToNumpy(image):

    imArray = image['composite']
    print(type(imArray))
    imArray = np.expand_dims(imArray, axis=-1)  # Kích thước trở thành (256, 256, 1)
    imArray = np.expand_dims(imArray, axis=0)
    imArray = tf.convert_to_tensor(imArray)
    resized_image = tf.image.resize(imArray, [28, 28])
    predict = model.predict(resized_image)
    return f"predict: {np.argmax(predict)}"

css = """
.gradio-container {
    display: flex;
    flex-direction: column;
    align-items: center;
}
.gradio-sketchpad {
    margin-bottom: 10px;
}
.gradio-button {
    margin-top: 5px;
}
"""
iface = gr.Interface(
    fn=sketchToNumpy,
    inputs=gr.Sketchpad(crop_size=(256, 256), type='numpy', image_mode='L', brush=gr.Brush()),
    outputs=gr.Textbox(),
    css=css
)

iface.launch()
