import string

        
def clean_title(title):
    return ''.join([word for word in title.lower() if word not in string.punctuation])