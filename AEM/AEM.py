"""
Terms:

(Current) Situation:
    - state + activity



TODO:
    - Think of representation for Episode

Problems:
    - everything is very abstract
    - Episode could be described in multiple messages, or multiple Episodes could be described in one message
        - simplify: every message is one Episode

Notes:
    - store Episode as spacy doc
    - use sentiment analysis to determine whether Episode is good or bad
    - use spacy to capture similiarities of memories/docs
        - incoorporate sentiment analysis into these similarities

    
    - Implement Episode retrieval (using spacy similarity + emotional)
        - Can reliably go back to a specific episode if we talk about something similar?
"""



class Episode:
    """
    Experiences/Situations encoded in a way that allows distances to other Episodes to be calculated
    """


class EmotionalQuality:
    """
    Emotional quality of an episode
    """



class EpisodicMemory:
    """
    Stores a user's personal events from their past, encoded as Episodes
    """




class GoalRepository:
    """
    Stores a user's existing goals/motivations and concerns
    """


em = EpisodicMemory()
gr = GoalRepository()




def ecphoricProcessing(state, activity):
    """
    Convert current situation into one or more Episodes
    Determines associative strength (distance) between that Episode and all other Episodes stored in Memory

    """
    em





def emotionalAppraisal(episode) -> EmotionalQuality:
    """
    Determine emotional quality of experiencing an Episode based on user's Goal Repository
    """
    gr


