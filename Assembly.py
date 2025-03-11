import argparse
import ast
import re
from MySQL import MySQL
from collections import defaultdict
from TaggerClassifier import TaggerClassifier
from TagClassifier import TagClassifier
from UserHistoryProcessor import UserHistoryProcessor
from StudentTaggingProcessor import StudentTaggingProcessor
from KrippendorffAlphaProcessor import KrippendorffAlphaProcessor
from AgreementDisagreementProcessor import AgreementDisagreementProcessor
from PatternDetectionProcessor import PatternDetectionProcessor
from IntervalLogsHandler import IntervalLogsHandler
from PatternDetection_refactored import PatternDetection
import numpy as np
import pandas as pd
import os
import csv
import ast

class Application:
    """
    Master class of the Application
    """
    def __init__(self) -> None:
        self._connector = MySQL()                       # MySQL connector to call methods of MySQL class
        #self.assignment_to_users = defaultdict(dict)    # dictionary to store the result of interval logs query
        self.tagger_classifier = TaggerClassifier()     # object of TaggerClassifier class
        self.interval_logs_handler = IntervalLogsHandler(self.tagger_classifier)  # IntervalLogsHandler instance
        self.tag_classifier = TagClassifier()           # object of TagClassifier class
       # self.interval_logs_result = defaultdict(dict)   # result of interval logs
        self.krippendorff_result = defaultdict(dict)    # result of krippendorff alpha for a user
        self.agree_disagree_tags = defaultdict(dict)    # result of agreement/disagreement for each tag
        self.pattern_detection_result = defaultdict(dict) # result of interval logs
        self.assignment_to_user = defaultdict(dict)   
        self.user_history_dict = defaultdict(dict)
        self.pattern_detection = PatternDetection()
        self.assignment_to_teams = {}                  # dictionary that stores the result of getUserTeams function
        self.user_history_processor = UserHistoryProcessor(self.tagger_classifier)
        self.student_tagging_processor = StudentTaggingProcessor()
        self.krippendorff_processor = KrippendorffAlphaProcessor(self.tagger_classifier)
        self.agreement_processor = AgreementDisagreementProcessor(self.tag_classifier)
        self.pattern_processor = PatternDetectionProcessor(self.pattern_detection)

    def __getUserHistory(self, user_history) -> None:
        """Calls UserHistoryProcessor to handle user history processing."""
        self.user_history_processor.process_user_history(user_history)
  
    def __getStudentsWhoTagged(self) -> None:
        """Calls StudentTaggingProcessor to process and generate tagging data."""
        self.student_tagging_processor.process_tagging_data()

    def __getKrippendorffAlpha(self, alpha=None) -> None:
        """Calls KrippendorffAlphaProcessor to compute Krippendorff's alpha."""
        self.krippendorff_processor.compute_alpha(self.assignment_to_teams, alpha)
     
    def __calculateAgreementDisagreement(self) -> None:
        """Calls AgreementDisagreementProcessor to compute agreement/disagreement."""
        self.agreement_processor.compute_agreement_disagreement(self.assignment_to_teams)
        
    def assignTaggerReliability(self, log_time=None, alpha=None, lmin=5, lmax=30, minrep=15):
        """
        Computes interval logs, Krippendorff alpha, and pattern detection.
        """
        # Interval logs
        tags = self._connector.getAnswerTags()
        self.interval_logs_handler.calculate_interval_logs(tags, log_time)
        
    def assignTagReliability(self):
        """
        Function used to compute Agreement/Disagreement of tags
        """
        self.__calculateAgreementDisagreement()


    def __getPatternResults(self, tags, lmin=5, lmax=30, minrep=15) -> None:
        """Calls PatternDetectionProcessor to process patterns."""
        self.pattern_processor.process_patterns(tags, self.assignment_to_user, lmin, lmax, minrep)
    
    def calculate_credibility(self, log_time, alpha, total_characters, log_time_max, alpha_max, characters_max):

        # Check and handle non-numeric or missing values
        log_time = 0 if not isinstance(log_time, (int, float)) or pd.isna(log_time) else log_time
        alpha = 0 if not isinstance(alpha, (int, float)) or pd.isna(alpha) else alpha
        total_characters = 0 if not isinstance(total_characters, (int, float)) or pd.isna(total_characters) else total_characters

        normalized_log_time = log_time / log_time_max if log_time_max != 0 else 0
        normalized_alpha = alpha / alpha_max if alpha_max != 0 else 0
        normalized_characters = 1 - (total_characters / characters_max if characters_max != 0 else 0)
        
        credibility = (normalized_log_time + normalized_alpha + normalized_characters) / 3
        return round(credibility, 5)
    

    def remove_indices_smaller(self, pattern, rep):
        indices_to_remove = [i for i, p in enumerate(pattern) if len(set(p))<2]
        return [r for i, r in enumerate(rep) if i not in indices_to_remove]

    def update_results(self, path):
        df = pd.read_csv(path)
        df['Pattern'] = df['Pattern'].fillna(df['Pattern'].apply(lambda x: []))
        df['Pattern Repetition'] = df['Pattern Repetition'].fillna(df['Pattern Repetition'].apply(lambda x: []))
        df['Pattern Repetition'] = df['Pattern Repetition'].apply(lambda x: eval(x) if isinstance(x, str) else x)
        df['Pattern'] = df['Pattern'].apply(lambda x: eval(x) if isinstance(x, str) else x)
        df['Pattern Repetition'] = df.apply(lambda row: self.remove_indices_smaller(row['Pattern'], row['Pattern Repetition']), axis=1)
        df['Pattern'] = df['Pattern'].apply(lambda x:[item for item in x if len(set(item)) > 1])
        df['Total Repeating Characters'] = df['Pattern Repetition'].apply(lambda arr: int(np.sum(arr)))
        
        # Check for pattern existence and create new column
        df['Pattern Found or Not'] = df['Pattern'].apply(lambda patterns: 'Found' if any(patterns) else 'Not Found')
    
        return df

    def process_and_save_final_results(self, results_filename, longest_yn_filename, output_filename):
            # Read CSV files
            long_y_n = pd.read_csv(longest_yn_filename)
            df = pd.read_csv(results_filename)
            
            # Merge DataFrames
            merged_df = df.merge(long_y_n, left_on='User ID', right_on='User')
            
            # Reorder columns and rename Credibility column
            cred_column = merged_df.pop("Credibility")
            merged_df["Credibility"] = cred_column
            merged_df.drop(columns=["User", "Tags"], inplace=True)

            # Save to CSV
            merged_df.to_csv(output_filename, index=False)

            print(f"Final file combining all the results is written at {output_filename}")



    def combine_csv_results(self, output_file):
        # Read the CSV files into DataFrames
        interval_logs_df = pd.read_csv("data/Interval_logs.csv")
        krippendorff_df = pd.read_csv("data/krippendorff.csv")
        pattern_results_df = pd.read_csv("data/Pattern_recognition.txt", sep="/", header=0, names=["Assignment_id", "User_id", "PD_result", "Pattern", "Repetition"])

        # Merge the DataFrames on 'Assignment_id' and 'User_id'
        merged_df = interval_logs_df.merge(krippendorff_df, on=['Assignment_id', 'User_id'])
        merged_df = merged_df.merge(pattern_results_df, on=['Assignment_id', 'User_id'])

        # Ensure the 'Time' column is present after the merge
        print(merged_df.columns)

        # Replace 'N/A' with a space in the entire DataFrame
        merged_df.replace('N/A', ' ', inplace=True)
        
        # Replace '-1' with a space in the 'Fast Tagging Log Values' column
        merged_df['IL_result'] = merged_df['IL_result'].apply(lambda x: ' ' if x == -1 else x)

        # Replace pattern strings like "('-1', '1', '1', '-1', '1', '1')" with "NYYNYY"
        merged_df['Pattern'] = merged_df['Pattern'].apply(
            lambda x: ''.join(['Y' if num == '1' else 'N' for num in ast.literal_eval(x)]) 
                    if isinstance(x, str) and x.startswith('(') else x
        )
            
        # Adding the 'Number of Tags Available' column using 'team_id'
        merged_df['Number_of_Tags_Available'] = merged_df['Team_id'].apply(lambda team_id: self._connector.getAnswerCount(team_id))


        # Adjust the column order and rename as needed
        merged_df = merged_df[['User_id', 'Assignment_id', 'Team_id', 'IL_result', 'Time', 'Alphas', "Number_of_Tags", "Number_of_Tags_Available",  'PD_result', 'Pattern', 'Repetition']]
        merged_df.columns = ['User ID', 'Assignment ID', 'Team ID', 'Fast Tagging Log Values', 'Fast Tagging Seconds', 'Alpha Values', 'Number of Tags Set', 'Number of Tags Available', 'Pattern Found or Not', 'Pattern', 'Pattern Repetition']

        #Ensuring that patterns of a single user appear in the same line
        agg_funcs = {'Assignment ID': 'min','Team ID':'min','Fast Tagging Log Values' :'min', 'Fast Tagging Seconds':'min','Alpha Values':'min','Number of Tags Set':'min',
             'Number of Tags Available':'min','Pattern Found or Not':'first', 'Pattern': lambda x: x.tolist(),'Pattern Repetition': lambda x: x.tolist()}
        # Group by 'id' and aggregate selected columns
        result_df = merged_df.groupby('User ID', as_index=False).agg(agg_funcs)

        #removing 'nan' values for users where no pattern was found
        rows_not_found = result_df[result_df['Pattern Found or Not'] == 'Not_found'].index
        columns_to_blank = ['Pattern', 	'Pattern Repetition']  
        result_df.loc[rows_not_found, columns_to_blank] = ''

        #removing single quotes for each pattern found inside the array
        result_df['Pattern'] = result_df['Pattern'].replace({'["\']': ''}, regex=True) 

        #function that calculates the result of pattern length * pattern repetition
        def calculate_score(row):
            # Split the columns
            patterns = row['Pattern']
            repetitions = row['Pattern Repetition']

            # Check if 'Pattern' and 'Pattern Repetition' are not None and not empty lists
            if patterns is not None and repetitions is not None and patterns and repetitions:
                total_score = sum(len(pattern) * repetition for pattern, repetition in zip(patterns, repetitions))
                return total_score if total_score != 0 else None
            else:
                return None        

        # Apply the function 
        result_df['Total Repeating Characters'] = result_df.apply(calculate_score, axis=1)

        # Handling any Nan values or cases where no pattern is found
        rows = result_df[result_df['Pattern Found or Not'] == 'Not_found'].index
        result_df.loc[rows, 'Total Repeating Characters'] = 0  # Set to 0 instead of empty string


         # Find the maximum values for normalization
        log_time_max = result_df['Fast Tagging Seconds'].max()
        alpha_max = result_df['Alpha Values'].max()
        characters_max = result_df['Total Repeating Characters'].max()

        # Calculate credibility for each row
        result_df['Credibility'] = result_df.apply(lambda row: self.calculate_credibility(
            row['Fast Tagging Seconds'], row['Alpha Values'], row['Total Repeating Characters'],
            log_time_max, alpha_max, characters_max
        ), axis=1)

        # Adjust the column order and rename as needed
        result_df = result_df[['User ID', 'Assignment ID', 'Team ID', 'Fast Tagging Log Values', 'Fast Tagging Seconds', 'Alpha Values', 'Number of Tags Set', 'Number of Tags Available', 'Pattern Found or Not', 'Pattern', 'Pattern Repetition','Total Repeating Characters', 'Credibility']]

        # Write the combined results to a new CSV file
        output_path = f"data/{output_file}"
        result_df.to_csv(output_path, index=False, na_rep=' ')

        # Update the result files for non-consecutive patterns
        result_df = self.update_results(output_path)
        result_df.to_csv(output_path, index=False, na_rep=' ')


        print(f"Combined CSV created successfully as {output_path}")

        self.process_and_save_final_results("data/Combined_Results.csv", "data/Longest_Y_N.csv", "data/1166_Tagger_Results.csv")


if __name__ == "__main__":
    # Ensure data directory exists
    os.makedirs('data', exist_ok=True)
    parser = argparse.ArgumentParser(description="Run the Application with specific parameters.")
    parser.add_argument('--log_time_min', type=float, default=None, help="Filtering value for log time.")
    parser.add_argument('--alpha_min', type=float, default=None, help="Filtering value for krippendorff alpha.")
    parser.add_argument('--min_pattern_len', type=int, default=10, help="Minimum value for pattern detection.")
    parser.add_argument('--max_pattern_len', type=int, default=50, help="Maximum value for pattern detection.")
    parser.add_argument('--min_pattern_rep', type=int, default=15, help="Minimum repetition value for pattern detection.")
    args = parser.parse_args()

    app = Application()
 
    app.assignTaggerReliability(args.log_time_min, args.alpha_min, args.min_pattern_len, args.max_pattern_len, args.min_pattern_rep)
    app.combine_csv_results('Combined_Results.csv')
