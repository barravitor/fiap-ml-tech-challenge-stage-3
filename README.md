# FIAP TECH CHALLENGE | Creating a Machine Learning Algorithm

## Índice

- [Introduction](#introduction)
- [Technologies Used](#technologies-used)
- [Installation and run](#installation-and-run)
- [Contribution](#contribution)
- [License](#license)
- [Contact](#contact)

## Introduction

Project responsible for training machine learning algorithms for Embrapa's wine data [link](http://vitibrasil.cnpuv.embrapa.br/index.php)

## Technologies Used

- **Python**: The project's main language, chosen for its rich library for data analysis.
- **scikit-learn**: Library used for statistical modeling and machine learning, including algorithms such as regression, classification, preprocessing, cross-validation, and performance evaluation.
- **numpy**: Fundamental library for numerical computing. Used for high-performance vector operations, matrices and mathematical functions.
- **pandas**: Library used for manipulation and analysis of tabular data, such as reading CSVs, filtering, grouping and data transformations.

## Installation and run

Instructions on how to install and run the project.

Create a .env file in the project root following the example in the .env-example file

Required python version: 3.10.12

```bash
python3 -m venv .venv # Run to create the environment
source .venv/bin/activate # Run to start the environment
pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt # Run to install the necessary packages
```

Run the train script to generate model
```bash
python -m training.train_productions_model
```

Run the prediction script test model
```bash
python -m prediction.predict_productions
```

Run the API to load predict data
```bash
uvicorn api.app.main:app --host 0.0.0.0 --port 8000 --reload # Run in dev mode
```

## Contribution

We welcome contributions to this project! Here’s how you can help:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/YourFeatureName`).
3. Make your changes and commit them (`git commit -m 'feat: Add some feature'`).
4. Push to the branch (`git push origin feature/YourFeatureName`).
5. Open a Pull Request.

Please ensure that your code adheres to the project's coding standards and includes appropriate tests where necessary.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE.txt) file for details.

## Contact

For questions, suggestions, or feedback, please contact:

* **Edson Vitor**  
  GitHub: [barravitor](https://github.com/barravitor)