import pandas as pd
import csv

class StudentTaggingProcessor:
    """
    Handles processing of student tagging data.
    """

    def __init__(self, input_file="data/user_data.csv", output_file="data/number_of_students_who_tagged_each_question.csv"):
        self.input_file = input_file
        self.output_file = output_file

    def process_tagging_data(self):
        """Reads user data, calculates unique students per question, and writes to a CSV file."""
        df = pd.read_csv(self.input_file, encoding='cp1252')

        # Count unique users per question
        result_df = df.groupby('Question')['User_id'].nunique().reset_index()
        result_df.columns = ['Question', 'Unique_User_Count']

        # Write results to CSV
        result_df.to_csv(self.output_file, index=False, na_rep=' ', quoting=csv.QUOTE_MINIMAL)

        print(f"Results saved to '{self.output_file}'")
