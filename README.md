# Bee and BumbleBee Detector

This repository contains the code and resources for the project **Bee and BumbleBee Detector**, which applies classical machine learning and advanced deep learning techniques to detect bee sounds in natural soundscapes. The project aims to support ecological conservation efforts by offering a reliable and non-invasive method for monitoring bee populations through acoustic data.

## Project Overview

Bees play a crucial role in biodiversity, pollination, and agricultural productivity, making their monitoring vital for conservation strategies. Traditional monitoring techniques can disrupt natural bee behaviors, which is why this project focuses on **acoustic monitoring**—a non-invasive and cost-effective approach.

However, acoustic monitoring presents challenges due to the complexity of natural soundscapes. This project tackles these challenges by evaluating various audio classification techniques and introducing **end-to-end deep learning models** to improve the accuracy and reliability of bee sound detection. By automating feature extraction, the models enhance performance compared to traditional machine learning methods.

### Key Features
- **Binary Classification**: Detects and classifies short audio frames into 'Bee' and 'NoBee' classes.
- **Feature Engineering**: Evaluates multiple audio features such asMFCC, Spectral flatness, Spectral Centroid, Spectral Rolloff, Spectral Contrast, Chroma, STFT, Zero Crossing Rate,Constant-Q Transform, identifying the most effective combination for bee sound detection.
- **Classical Machine Learning Models**: Implements classifiers such as Logistic Regression K-Nearest Neighbor, Support Vector Machine, Extreme Gradient Boosting, Random Forest, Extremely Randomized Trees, K-Means using Librosa library.
- **Deep Learning Models**: Implements a classic Feed-Forward Neural Network and 1D CNN-based end-to-end approach that surpasses traditional models like Gradient Boosting, providing enhanced consistency and performance.
- **Scalability**: Demonstrates the robustness of deep learning models with varying training data sizes, confirming their potential for large-scale acoustic monitoring.

## Insights from the Study

- **Optimal Feature Set**: Spectral flatness, zero crossing rate, spectral roll-off, MFCC, spectral contrast, and chroma were identified as the most effective features for representing bee sounds. These features outperformed other combinations in distinguishing bee audio from background noise.
  
- **Machine Learning Models**: Gradient Boosting achieved the best performance among traditional machine learning methods. However, the study found that the 1D CNN end-to-end model outperformed all traditional methods by automating feature extraction and refining raw audio inputs.

- **Impact of Data Size**: The study revealed that larger training datasets improve model performance. The end-to-end model demonstrated a linear improvement in accuracy with increased data, suggesting that expanding the dataset can further enhance detection capabilities.

- **Cross-validation**: A 5-fold cross-validation process confirmed the reliability and robustness of the end-to-end model, even with varying training data sizes.

## Repository Structure

```
Bee-BumbleBee-Detector/
│
├── data/                 # Contains the bee sound datasets (raw audio files)
├── models/               # Pre-trained and trained models for sound classification
├── notebooks/            # Jupyter notebooks for data exploration and model training
├── scripts/              # Python scripts for preprocessing, training, and evaluation
├── results/              # Model performance metrics and result visualizations
├── requirements.txt      # List of dependencies for the project
└── README.md             # Project documentation (this file)
```

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/rozhin2025/Bee-BumbleBee-Detector.git
    cd Bee-BumbleBee-Detector
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. **Data Preprocessing**: Use the provided scripts in the `scripts/` folder to preprocess raw audio data before training.
    ```bash
    python scripts/preprocess_data.py
    ```

2. **Train the Model**: Train the deep learning model using the training script.
    ```bash
    python scripts/train_model.py --epochs 50 --batch_size 32
    ```

3. **Evaluate the Model**: Evaluate the model's performance using test data.
    ```bash
    python scripts/evaluate_model.py
    ```

4. **Visualize Results**: Jupyter notebooks in the `notebooks/` folder can be used to visualize the results and performance metrics of the model.

## Results

The project highlights that properly configured end-to-end deep learning models surpass traditional machine learning approaches like Gradient Boosting in detecting bee sounds. These models provide reliable, scalable solutions for non-invasive bee monitoring, offering promising avenues for conservation strategies.

## Future Work and Limitations

- **Method Exploration**: Future research can explore additional machine learning methods and audio features to further improve classification accuracy.
- **Dataset Expansion**: Expanding the training dataset could enhance model performance, particularly in understanding the optimal data size for improved generalization.
- **Recording Quality**: Higher-quality recordings, especially in noisy urban environments, may improve accuracy and generalization across diverse acoustic environments.

## Technologies

- **Python**
- **TensorFlow/Keras** for deep learning
- **Librosa** for audio data processing
- **Matplotlib/Seaborn** for data visualization

## References

### References

This project draws inspiration from two key papers in the field of audio classification:

1. **Phan, T.-T.-H., Nguyen, H.-D., & Nguyen, D.-D. (2022). Evaluation of feature extraction methods for bee audio classification. Lecture Notes on Data Engineering and
Communications Technologies, 148. https://doi.org/10.1007/978-3-030-92237-4_36**: This study offers a comprehensive comparison of six machine learning methods (Logistic Regression, KNN, SVM, XGBoost, Random Forest, and Extremely Randomized Trees) evaluated across five audio features, including Mel Frequency Cepstral Coefficient (MFCC) and Chroma. The findings emphasize that MFCC and Chroma features are particularly effective for representing bee sounds and enhancing the performance of classification models.

2. **Abdoli, S., Cardinal, P., & Koerich, A. L. (2019). End-to-end environmental sound classification using a 1d convolutional neural network. Expert Systems with Applications, 130, 73–81. https://doi.org/10.1016/j.eswa.2019.06.040**: This paper focuses on classifying environmental sounds using a 1D Convolutional Neural Network (CNN) in an end-to-end approach, bypassing the need for handcrafted features like MFCC. The 1D CNN learns to extract relevant audio features directly from the raw data, outperforming traditional 2D representations and CNN-based methods.

## Contributions

Contributions are welcome! Feel free to submit issues or pull requests if you have suggestions for improvements or additional features.
