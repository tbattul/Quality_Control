import numpy as np
import csv
from collections import defaultdict

class KrippendorffAlphaProcessor:
    """
    Handles the computation of Krippendorff's alpha for tagging reliability.
    """
    def __init__(self, tagger_classifier):
        self.tagger_classifier = tagger_classifier
        self.krippendorff_result = defaultdict(dict)

    def compute_alpha(self, assignment_to_teams, alpha=None):
        """Calculates Krippendorff's alpha for each user and writes results to a CSV file."""
        for assignment in assignment_to_teams:
            for team in assignment_to_teams[assignment].teams:
                data, answers, users = self.__collect_team_data(assignment_to_teams, assignment, team)

                if not data or len(data[0]) == 1:
                    self.krippendorff_result[assignment][team] = {users[0]: np.nan}
                    continue

                data = np.array(data)
                self.krippendorff_result[assignment][team] = self.tagger_classifier.computeKrippendorffAlpha(data, users)

        self.__write_results_to_csv(alpha)

    def __collect_team_data(self, assignment_to_teams, assignment, team):
        """Gathers tagging data for Krippendorff's alpha computation."""
        data, answers, users = [], defaultdict(set), []

        for user in assignment_to_teams[assignment].teams[team].users:
            users.append(user)
            for answer in assignment_to_teams[assignment].teams[team].users[user].answers:
                for tag in assignment_to_teams[assignment].teams[team].users[user].answers[answer].tags:
                    answers[answer].add(tag)

        for answer in answers:
            for tag in answers[answer]:
                row = [
                    self.__get_tag_value(assignment_to_teams, assignment, team, user, answer, tag)
                    for user in assignment_to_teams[assignment].teams[team].users
                ]
                data.append(row)

        return data, answers, users

    def __get_tag_value(self, assignment_to_teams, assignment, team, user, answer, tag):
        """Retrieves the tag value or assigns NaN if missing."""
        user_answers = assignment_to_teams[assignment].teams[team].users[user].answers
        if answer not in user_answers or tag not in user_answers[answer].tags:
            return np.nan
        return user_answers[answer].tags[tag].value

    def __write_results_to_csv(self, alpha):
        """Writes Krippendorff's alpha results to a CSV file."""
        with open("data/krippendorff.csv", "w") as f:
            f.write("Assignment_id,Team_id,User_id,Alphas\n")
            for assignment_id, teams in self.krippendorff_result.items():
                for team_id, users_alphas in teams.items():
                    for user_id, alpha_value in users_alphas.items():
                        if alpha is None or alpha_value >= alpha:
                            alpha_formatted = "{:.3f}".format(alpha_value) if isinstance(alpha_value, float) else "nan"
                            f.write(f"{assignment_id},{team_id},{user_id},{alpha_formatted}\n")
        print("Krippendorff's alpha written to data/krippendorff.csv")
