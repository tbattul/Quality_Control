from collections import defaultdict
import numpy as np
from TaggerClassifier import TaggerClassifier


class IntervalLogsHandler:
    def __init__(self, tagger_classifier):
        """
        Initializes the IntervalLogsHandler with a TaggerClassifier instance.
        """
        self.tagger_classifier = tagger_classifier
        self.assignment_to_users = defaultdict(dict)  # Stores tags grouped by assignment and user
        self.interval_logs_result = defaultdict(dict)  # Stores interval logs results

    def calculate_interval_logs(self, tags, log_time=None):
        """
        Calculates interval logs for the given tags.

        Args:
            tags (list): List of tags.
            log_time (float, optional): Minimum log time threshold. Defaults to None.
        """
        # Group tags by assignment and user
        for tag in tags:
            if tag.assignment_id in self.assignment_to_users:
                if tag.user_id in self.assignment_to_users[tag.assignment_id]:
                    self.assignment_to_users[tag.assignment_id][tag.user_id].append(tag)
                else:
                    self.assignment_to_users[tag.assignment_id][tag.user_id] = [tag]
            else:
                self.assignment_to_users[tag.assignment_id] = {tag.user_id: [tag]}

        # Calculate interval logs for each assignment and user
        for assignment_id, users in self.assignment_to_users.items():
            for user, tags in users.items():
                self.interval_logs_result[assignment_id][user] = self.tagger_classifier.buildIntervalLogs(
                    self.assignment_to_users[assignment_id][user]
                )

        # Write results to CSV
        self._write_interval_logs_to_csv(log_time)

    def _write_interval_logs_to_csv(self, log_time):
        """
        Writes interval logs results to a CSV file.

        Args:
            log_time (float, optional): Minimum log time threshold. Defaults to None.
        """
        with open("data/Interval_logs.csv", "w") as f:
            f.write("Assignment_id,User_id,IL_result,Time,Number_of_Tags\n")
            for assignment_id, users in self.interval_logs_result.items():
                for user_id, results in users.items():
                    log_time_value = results[0]
                    number_of_tags = results[1]
                    if log_time is None or log_time_value >= log_time:
                        il_result_formatted = "{:.3f}".format(log_time_value)
                        time_formatted = "{:.3f}".format(pow(2, log_time_value))
                        f.write(f"{assignment_id},{user_id},{il_result_formatted},{time_formatted},{number_of_tags}\n")
        print("Interval logs written to data/Interval_logs.csv")