import argparse

import openai


def create_context(
    question, search_file_id, max_len=1800, search_model="ada", max_rerank=10
):
    """
    Create a context for a question by finding the most similar context from the search file.
    :param question: The question
    :param search_file_id: The file id of the search file
    :param max_len: The maximum length of the returned context (in tokens)
    :param search_model: The search model to use
    :param max_rerank: The maximum number of reranking
    :return: The context
    """
    results = openai.Engine(search_model).search(
        search_model=search_model,
        query=question,
        max_rerank=max_rerank,
        file=search_file_id,
        return_metadata=True,
    )
    returns = []
    cur_len = 0
    for result in results["data"]:
        cur_len += int(result["metadata"]) + 4
        if cur_len > max_len:
            break
        returns.append(result["text"])
    return "\n\n###\n\n".join(returns)


def answer_question(
    question,
    max_len=1800,
    search_model="ada",
    max_rerank=10,
    debug=False,
    stop_sequence=["\n", "."],
    max_tokens=100,
):
    search_file_id = 'file-LfKVMSLJCVjmWHzrCG1eVREZ'
    #fine_tuned_qa_model = 'curie:ft-personal-2022-07-04-15-50-49'
    fine_tuned_qa_model = 'text-davinci-002'
    context = create_context(
        question,
        search_file_id,
        max_len=max_len,
        search_model=search_model,
        max_rerank=max_rerank,
    )
    if debug:
        print("Context:\n" + context)
        print("\n\n")
    try:
        # fine-tuned models requires model parameter, whereas other models require engine parameter
        model_param = (
            {"model": fine_tuned_qa_model}
            if ":" in fine_tuned_qa_model
            and fine_tuned_qa_model.split(":")[1].startswith("ft")
            else {"engine": fine_tuned_qa_model}
        )
        response = openai.Completion.create(
            prompt=f"Answer the question based on the context below\n\nText: {context}\n\n---\n\nQuestion: {question}\nAnswer:",
            temperature=0,
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=stop_sequence,
            **model_param,
        )
        res = response["choices"][0]["text"].strip()
        if not res.endswith('.'):
            res += '.'
        return res
    except Exception as e:
        print(e)
        return ""



if __name__ == '__main__':
    SIMPLE_QUESTIONS = [
        'Who is the partner in Deloitte legal?',
        'Who is the partner for consulting in Deloitte?',
        'Who is the partner in the Regulatory & Compliance department at Deloitte Legal in Prague?',
        'Who is the partner for Tax & Legal?',        
    ]

    MORE_COMPLEX_QUESTIONS = [
        'Name three partners in Deloitte.',
        'What is Jaroslava Kračúnová\'s role at Deloitte Legal?',
        'Who is responsible for M&A and delivery of financial due diligence services and what is his role?',
        'Who are the partners for consulting?',
        'Who should I contact regarding M&A?',
        'Who worked for Deloitte in New York and when?'
    ]


    print('''
██████╗░███████╗██╗░░░░░░█████╗░██╗████████╗████████╗███████╗
██╔══██╗██╔════╝██║░░░░░██╔══██╗██║╚══██╔══╝╚══██╔══╝██╔════╝
██║░░██║█████╗░░██║░░░░░██║░░██║██║░░░██║░░░░░░██║░░░█████╗░░
██║░░██║██╔══╝░░██║░░░░░██║░░██║██║░░░██║░░░░░░██║░░░██╔══╝░░
██████╔╝███████╗███████╗╚█████╔╝██║░░░██║░░░░░░██║░░░███████╗
╚═════╝░╚══════╝╚══════╝░╚════╝░╚═╝░░░╚═╝░░░░░░╚═╝░░░╚══════╝

██╗░░██╗███╗░░██╗░█████╗░░██╗░░░░░░░██╗██╗░░░░░███████╗██████╗░░██████╗░███████╗  ██████╗░░█████╗░░██████╗███████╗
██║░██╔╝████╗░██║██╔══██╗░██║░░██╗░░██║██║░░░░░██╔════╝██╔══██╗██╔════╝░██╔════╝  ██╔══██╗██╔══██╗██╔════╝██╔════╝
█████═╝░██╔██╗██║██║░░██║░╚██╗████╗██╔╝██║░░░░░█████╗░░██║░░██║██║░░██╗░█████╗░░  ██████╦╝███████║╚█████╗░█████╗░░
██╔═██╗░██║╚████║██║░░██║░░████╔═████║░██║░░░░░██╔══╝░░██║░░██║██║░░╚██╗██╔══╝░░  ██╔══██╗██╔══██║░╚═══██╗██╔══╝░░
██║░╚██╗██║░╚███║╚█████╔╝░░╚██╔╝░╚██╔╝░███████╗███████╗██████╔╝╚██████╔╝███████╗  ██████╦╝██║░░██║██████╔╝███████╗
╚═╝░░╚═╝╚═╝░░╚══╝░╚════╝░░░░╚═╝░░░╚═╝░░╚══════╝╚══════╝╚═════╝░░╚═════╝░╚══════╝  ╚═════╝░╚═╝░░╚═╝╚═════╝░╚══════╝
    ''')

    print('SAMPLE QUESTIONS \n')

    print('    SIMPLE QUESTIONS:')
    for q in SIMPLE_QUESTIONS:
        print('     - ', q)
    print()

    print('    MORE COMPLEX QUESTIONS:')
    for q in MORE_COMPLEX_QUESTIONS:
        print('     - ', q)
    print()

    while True:
        question = input('Ask me any question... \nQ: ')
        ans = answer_question(question)
        print(f'A: {ans}')
        print()
