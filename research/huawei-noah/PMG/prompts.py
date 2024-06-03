
NEGATIVE_PROMPT = {
    'costume': 'human woman man body face fashion trendy',
    'movie': 'broken text, unclear, twisted', 
    'emoticon': 'Bad Artist, Worst Quality, Low Quality, Bad Hands-5, Low Resolution, Bad Anatomy, Bad Hands, Text, Watermark, Worst Quality, Low Quality, Blurry, Extra Limbs'
}

STYLE_PROMPT = {
    'costume':  "The assistant is helping the human generate keywords of shopper's interests.\
            ### Human: A shopper has bought some products.\
            Please provide <Keyword_num/> distinctive keywords in English that describe shoppers interests in theme, color, material, or appearance style, excluding specific product types such as skirts or pants.\
            The example of output is \"1. masculine 2. cute 3. cartoon ...\" ### Human: The description is \"<Items/>\". ###Assistant: The English keywords are ",
    'movie': "The assistant is helping the human generate keywords of movie lover's interests.\
             ### Human: A person watched some movies.\
             Please provide <Keyword_num/> keywords to describe his movie interests.\
             The example of output is \"The keywords are: 1. Keyword 1. 2. Keyword 2. 3. Keyword 3.\"\
             ### Human: The movies and their information are \"<Items/>\". \
             ### Assistant: The keywords are: "
}
STYLE_PROMPT_ATTRIBUTE = {
    'costume':  "The assistant is helping the human generate keywords of shopper's interests.\
            ### Human: A shopper has bought some products.\
            Please provide <Keyword_num/> distinctive keywords in English that describe shoppers interests about <Attribute/>, excluding specific product types such as skirts or pants.\
            The example of output is \"1. masculine 2. cute 3. cartoon ...\" ### Human: The description is \"<Items/>\". ###Assistant: The English keywords are ",
    'movie': "The assistant is helping the human generate keywords of movie lover's interests.\
             ### Human: A person watched some movies.\
             Please provide <Keyword_num/> keywords to describe his movie interests about <Attribute/>.\
             The example of output is \"The keywords are: 1. Keyword 1. 2. Keyword 2. 3. Keyword 3.\"\
             ### Human: The movies and their information are \"<Items/>\". \
             ### Assistant: The keywords are: "
}
STYLE_PROMPT_ATTRIBUTE_INTEGRATE = {
    'costume':  "The assistant is helping the human generate keywords of shopper's interests.\
            ### Human: A shopper has bought some products.\
            Here are some candidate keywords '<Candidate_keywords/>' \
            Please select <Keyword_num/> distinctive keywords from the candidate keywords.\
            The example of output is \"1. masculine 2. cute 3. cartoon ...\" ### Human: The description is \"<Items/>\". ###Assistant: The English keywords are ",
    'movie': "The assistant is helping the human generate keywords of movie lover's interests.\
             ### Human: A person watched some movies.\
             Here are some candidate keywords '<Candidate_keywords/>' \
             Please select <Keyword_num/> distinctive keywords from the candidate keywords.\
             The example of output is \"The keywords are: 1. Keyword 1. 2. Keyword 2. 3. Keyword 3.\"\
             ### Human: The movies and their information are \"<Items/>\". \
             ### Assistant: The keywords are: "
}

STYLE_PROMPT_SOFT = {
    'costume' : "### Human: A shopper has bought the following products: \"<Items/>\" .\
             Please describe his interests.\
             ###Assistant: ",
    'movie': "### Human: A person watched the following movies \"<Items/>\" .\
              Please describe his movie interests.\
              ### Assistant: "
}

ITEM_PROMPT = {
    'movie': "### Human: Moive title: \"<Title/>\"\n Movie introduction: \"<Intro/>\"\n Movie genre: \"<Genre/>\"\n Please design a poster for it (describe in 10 keywords without explaining). Character names should be excluded.\n The example of output is \"1. Keyword 1 2. Keyword 2 3. Keyword 3 ... \"\n ###Assistant: The keywords are: "
}

    