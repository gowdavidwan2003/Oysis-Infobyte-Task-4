# Oysis-Infobyte-Task-4
EMAIL SPAM DETECTION WITH MACHINE LEARNING

We’ve all been the recipient of spam emails before. Spam mail, or junk mail, is a type of email
that is sent to a massive number of users at one time, frequently containing cryptic
messages, scams, or most dangerously, phishing content.

In this Project, use Python to build an email spam detector. Then, use machine learning to
train the spam detector to recognize and classify emails into spam and non-spam.

# SMS Spam Detection

This project uses a Logistic Regression model to classify SMS messages as either 'spam' or 'ham' (not spam).

## Dependencies

The project requires the following Python libraries:
- numpy
- pandas
- matplotlib
- seaborn
- sklearn

## Dataset

The dataset used in this project is 'spam.csv'. It contains SMS messages and their corresponding labels ('spam' or 'ham').

## Usage

1. Import the necessary libraries.
2. Load the dataset using pandas.
3. Preprocess the data (encoding categorical variables, removing duplicates, and adding a new feature for the number of characters in each message).
4. Visualize the data using histograms.
5. Vectorize the text messages using CountVectorizer.
6. Split the data into training and testing sets.
7. Fit a Logistic Regression model using sklearn.
8. Evaluate the model's performance using accuracy score and classification report.
9. Predict labels for new data.

## Results

The Logistic Regression model provides a prediction of whether an SMS message is 'spam' or 'ham' based on its content. The model's performance is evaluated using accuracy score and a classification report.

## Future Work

This is a simple Logistic Regression model which assumes certain relationships between the dependent and independent variables. For future work, more complex models could be explored to capture any non-linear relationships in the data.
