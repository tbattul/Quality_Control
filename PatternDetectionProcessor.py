import numpy as np
import pandas as pd
import ast
import csv

class PatternDetectionProcessor:
    """
    Handles pattern detection in tagging data.
    The PatternDetectionProcessor.py performs the following tasks:

    Processes Patterns in Tagging Data – Organizes tags by assignment and user, detects patterns, and saves results.
    Detects Long Consecutive ‘Y’ and ‘N’ Sequences – Identifies and counts patterns longer than 10.
    Writes Pattern Detection Results – Saves processed data to user_tags.csv, Pattern_recognition.txt, and Longest_Y_N.csv.
    Cleans and Formats Tags – Converts tag values (1 → ‘Y’, -1 → ‘N’), ensuring consistent formatting.

    """

    def __init__(self, pattern_detection):
        self.pattern_detection = pattern_detection
        self.pattern_detection_result = {}

    def process_patterns(self, tags, assignment_to_user, lmin=5, lmax=30, minrep=15):
        """Processes patterns from tags and writes results to CSV files."""
        self.__populate_assignment_to_user(tags, assignment_to_user)
        self.__detect_patterns(assignment_to_user, lmin, lmax, minrep)
        self.__write_pattern_results_to_file()
        self.__find_Ys_Ns("data/user_tags.csv")

    def __populate_assignment_to_user(self, tags, assignment_to_user):
        """Organizes tag data by assignment and user."""
        for tag in tags:
            assignment_to_user.setdefault(tag.assignment_id, {}).setdefault(tag.user_id, []).append(tag)

    def __detect_patterns(self, assignment_to_user, lmin, lmax, minrep):
        """Runs pattern detection and writes user tagging data to a CSV."""
        for assignment_id, users in assignment_to_user.items():
            user_df = pd.DataFrame(columns=["User", "Tags"])
            for user, tags in users.items():
                tags.sort(key=lambda l: l.created_at)
                temp_bin_data = [i.value for i in tags]
                temp_user = pd.DataFrame({"User": user, "Tags": [temp_bin_data]})
                user_df = pd.concat([user_df, temp_user], ignore_index=True)

                pattern_results = self.pattern_detection.PTV(tags, lmin, lmax, minrep)
                self.pattern_detection_result.setdefault(assignment_id, {})[user] = pattern_results

            user_df.to_csv("data/user_tags.csv", index=False)
            print("User Tags are written to 'data/user_tags.csv'")

    def __write_pattern_results_to_file(self):
        """Writes pattern detection results to a file."""
        with open("data/Pattern_recognition.txt", "w") as f:
            f.write("Assignment_id/User_id/PD_result/Pattern/Repetition\n")
            for assignment_id, users in self.pattern_detection_result.items():
                for user, patterns in users.items():
                    for pattern, count in patterns:
                        status = "Found" if pattern != "Not_found" else "Not_found"
                        f.write(f"{assignment_id}/{user}/{status}/{pattern}/{count}\n")
        print("Pattern recognition results written to data/Pattern_recognition.txt")

    def __find_Ys_Ns(self, tags_file_path):
        """Reads a CSV file, processes 'Tags' column, and writes results to a new CSV file."""
        df = pd.read_csv(tags_file_path)
        df['Tags'] = df['Tags'].apply(self.__replace_tags)
        df['Consecutive Ys Pattern Count'] = df.apply(self.__find_long_consecutive, axis=1, key='Y')
        df['Consecutive Ns Pattern Count'] = df.apply(self.__find_long_consecutive, axis=1, key='N')
        df['Total Repeating Ys'] = df['Consecutive Ys Pattern Count'].apply(lambda arr: int(np.sum(arr)))
        df['Total Repeating Ns'] = df['Consecutive Ns Pattern Count'].apply(lambda arr: int(np.sum(arr)))

        df.to_csv('data/Longest_Y_N.csv', index=False)
        print("Consecutive Ys and Ns Pattern results written to 'data/Longest_Y_N.csv'")

    def __replace_tags(self, tag_list):
        """Replaces '1' with 'Y', '-1' with 'N', and removes other values."""
        tag_list = ast.literal_eval(tag_list)
        return ['Y' if tag == '1' else 'N' if tag == '-1' else None for tag in tag_list if tag in ['1', '-1']]

    def __find_long_consecutive(self, row, key):
        """Finds consecutive sequences of 'Y' or 'N' longer than 10."""
        substrings, current_length = [], 0
        for c in row["Tags"]:
            if c == key:
                current_length += 1
            else:
                if current_length > 10:
                    substrings.append(current_length)
                current_length = 0
        if current_length > 10:
            substrings.append(current_length)
        return substrings
