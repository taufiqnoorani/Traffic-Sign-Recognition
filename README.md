# Traffic-Sign-Recognition


This repository contains the code for a Traffic Sign Recognition (TSR) system that integrates weather classification (based on VGG19), image enhancement (based on EnhanceNet) and traffic sign recognition (YOLOv5). The system classifies images to determine if adverse conditions are present, enhances the images based on detected conditions, and then performs traffic sign recognition.

Unfortunately, the trained model files could not be uploaded to the repository due to GitHub’s change in file size limit of 100MB. As a result, you will need to train the models yourself. The model architectures and training scripts are included in the repository, allowing you to train them using your data.

## Requirements

To run the code, you will need to install the following packages:

- Python 3.8 or higher
- PyTorch (with appropriate CUDA or MPS support)
- OpenCV
- PIL (Pillow)
- NumPy
- torchvision
- pandas
- glob2
- tqdm

You can install the required packages using pip:

```bash
pip install torch torchvision opencv-python pillow numpy pandas glob2 tqdm
```

If you plan to use the YOLOv5 model, you may also need to clone the YOLOv5 repository:
```bash
git clone https://github.com/ultralytics/yolov5.git
cd yolov5
pip install -r requirements.txt
```

CUDA and MPS Support

If you are using a system with CUDA support (for NVIDIA GPUs) or MPS (Metal Performance Shaders for Apple devices), make sure to modify your model loading to utilize these technologies for optimal performance. Here’s a sample code snippet to select the appropriate device:
```python
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
```

Acknowledgments

Special thanks to the creators of YOLO, VGGNet, EnhanceNet, GTSRB and CURE-TSD.

Feel free to use any part of this to better fit your project!
