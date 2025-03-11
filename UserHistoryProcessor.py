import re
import csv

class UserHistoryProcessor:
    """
    Handles processing of user history, credibility score calculation, and CSV generation.
    """
    def __init__(self, tagger_classifier):
        self.tagger_classifier = tagger_classifier
        self.user_history_dict = {}

    def process_user_history(self, user_history):
        """Populates user history dictionary and calculates credibility scores."""
        self.__populate_user_history_dict(user_history)
        credibility_scores = self.tagger_classifier.calculate_tag_credibility_score(user_history)
        self.__write_user_data_csv(credibility_scores)

        print("User data with credibility scores written to data/user_data.csv")

    def __populate_user_history_dict(self, user_history):
        """Populates the user history dictionary."""
        for tag in user_history:
            self.user_history_dict.setdefault(tag.assignment_id, {}).setdefault(tag.user_id, []).append(tag)

    def __write_user_data_csv(self, credibility_scores):
        """Writes user data with credibility scores to a CSV file."""
        with open("data/user_data.csv", "w") as f:
            f.write("User_id,Assignment_id,Question,Score,Review_Comment,Tag_Prompt,Tag_Value,Credibility_Score\n")
            for assignment_id, users in self.user_history_dict.items():
                for user, tags in users.items():
                    for tag in tags:
                        cleaned_question = self.__clean_text(tag.question)
                        cleaned_comments = self.__clean_text(tag.comments)
                        tag_credibility_score = credibility_scores.get(tag.id, 0)

                        f.write(f"{user},{assignment_id},{cleaned_question},{tag.answer_score},{cleaned_comments},{tag.prompt},{tag.value},{tag_credibility_score}\n")

    def __clean_text(self, text: str) -> str:
        """Removes HTML tags, commas, and newlines from text."""
        return re.sub('<.*?>', '', text).replace(',', '').replace('\n', '').replace('\r', '')
