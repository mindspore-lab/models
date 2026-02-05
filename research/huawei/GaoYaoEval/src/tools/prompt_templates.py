NUMBER_SYSTEM_PROMPT = '从以下文本中提取出其提供的最终答案，答案为数值类型，注意！请仅返回答案。'
CHOICE_SYSTEM_PROMPT = '从以下文本中提取出其提供的答案，如A,B,C等，如果里面有(1)或者(2)的话，其对应的就是A,B等，注意！请仅返回答案（如A、B等）。'
TRUE_OR_FALSE_SYSTEM_PROMPT = '从以下文本中提取出其提供的答案，如true, false等。注意输出格式，请直接返回答案不要解释，例如：true'
COMMON_SYSTEM_PROMPT = "You are a helpful assistant."

WIN_RATE_SYSYTEM_PROMPT = """[System]\n
        Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to
        the user question displayed below. You should choose
        the assistant that follows the user’s instructions and answers the user’s question better. Your evaluation should
        consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their
        responses. Begin your evaluation by comparing the
        two responses and provide a short explanation. Avoid
        any positional biases and ensure that the order in which
        the responses were presented does not influence your
        decision. Do not allow the length of the responses to
        influence your evaluation. Do not favor certain names
        of the assistants. Be as objective as possible. After
        providing your explanation, output your final verdict by
        strictly following this format: “[[A]]” if assistant A is
        better, “[[B]]” if assistant B is better, and “[[C]]” for a
        tie."""

MISSING_POINTS_SYSTEM_PROMPT = """
你是一个严格的评分专家，请根据下面规则给出两个分数。

问题：[question]
参考答案（视为若干得分点的合集）：[reference]
模型回答：[model_output]

请按以下步骤评分：

1. 从参考答案中提取若干“得分点”（关键事实或论证要点），总数不超过 6 个。
2. 判断模型回答中，哪些得分点被完整覆盖，哪些只是部分提及，哪些完全缺失：
   - "covered"：该点内容清楚、准确地出现。
   - "partially_covered"：有提到但不完整或有明显偏差。
   - "missing"：没有提及该点。
3. 计算缺失点数 missing_points：
   - 每一个 "missing" 记为 1 个缺失点；
   - 每一个 "partially_covered" 记为 0.5 个缺失点。
4. 以 100 分为满分，每 1 个缺失点扣 15 分：
   rubric_score = 100 - 15 * missing_points
   然后将 rubric_score 限制在 0 到 100 之间。
5. 此外，单独评估“整体语义相似度”（similarity），0-100 分，表示模型回答和参考答案在整体含义上的接近程度。

输出要求（非常重要）：
- 只输出一个 JSON 对象，不要任何解释文字，不要额外文本。
- JSON 格式必须是：
  {
    "points": [
      {"description": "要点1", "status": "covered"},
      {"description": "要点2", "status": "missing"},
      ...
    ],
    "similarity": 78
  }
"""