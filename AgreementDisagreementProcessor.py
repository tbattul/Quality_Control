import numpy as np
import csv
from collections import defaultdict

class AgreementDisagreementProcessor:
    """
    Handles the calculation of agreement and disagreement for tagging reliability.
    """
    def __init__(self, tag_classifier):
        self.tag_classifier = tag_classifier
        self.agree_disagree_tags = defaultdict(dict)

    def compute_agreement_disagreement(self, assignment_to_teams):
        """Calculates agreement/disagreement for each team and writes results to a CSV file."""
        for assignment in assignment_to_teams:
            for team in assignment_to_teams[assignment].teams:
                data, answers = self.__collect_team_data(assignment_to_teams, assignment, team)

                if not data or len(data[0]) == 0:
                    self.agree_disagree_tags[assignment][team] = np.nan
                    continue

                data = np.array(data)
                self.agree_disagree_tags[assignment][team] = self.tag_classifier.calculateAgreementDisagreement(data)

        self.__write_results_to_csv()

    def __collect_team_data(self, assignment_to_teams, assignment, team):
        """Gathers tagging data for agreement/disagreement calculation."""
        data, answers = [], defaultdict(set)

        for user in assignment_to_teams[assignment].teams[team].users:
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

        return data, answers

    def __get_tag_value(self, assignment_to_teams, assignment, team, user, answer, tag):
        """Retrieves the tag value or assigns None if missing."""
        user_answers = assignment_to_teams[assignment].teams[team].users[user].answers
        if answer not in user_answers or tag not in user_answers[answer].tags:
            return None
        return user_answers[answer].tags[tag]

    def __write_results_to_csv(self):
        """Writes agreement/disagreement results to a CSV file."""
        with open("data/tags.csv", "w") as f:
            f.write("Assignment_id,team_id,answer_id,tag_prompt_id,value,fraction\n")
            for assignment_id, teams in self.agree_disagree_tags.items():
                for team_id, answers in teams.items():
                    for answer_id, tags in answers.items():
                        for tag_id, values in tags.items():
                            f.write(f"{assignment_id},{team_id},{answer_id},{tag_id},{values[0]},{values[1]}\n")
        print("Agreement/Disagreement results written to data/tags.csv")
