from decimal import Decimal
from src.tools.file_operations import write_excel

def calc_acc_by_language(stats):
    """通用语言准确率计算"""
    langs, totals, matches, accs = [], [], [], []
    for lang, total in stats["total"].items():
        match = stats["match"].get(lang, 0)
        acc = float((Decimal(match) / Decimal(total)).quantize(Decimal('0.0001')))
        langs.append(lang)
        totals.append(total)
        matches.append(match)
        accs.append(acc)

    return {"language": langs, "accuracy": accs}

def export_country_acc_result(all_dict, match_dict):
    """
            从给定的字典中提取语言、行数、匹配数和准确率，并返回这些数据的列表。

            Args:
                language_to_line_num_dict: 一个字典，键为语言名称，值为该语言的行数。
                language_to_match_num_dict: 一个字典，键为语言名称，值为该语言的匹配数。
            Returns:
                accuracy_list: 每种语言的准确率列表。
                language_list: 语言名称列表。
                line_num_list: 每种语言的行数列表。
                match_num_list: 每种语言的匹配数列表。
            """
    language_list, country_list, line_num_list, match_num_list, accuracy_list = [], [], [], [], []
    language_country_dict = dict()
    for language in all_dict.keys():
        all_language_value = all_dict.get(language)
        match_language_value = match_dict.get(language, dict())
        country_dict_value = language_country_dict.get(language, dict())
        for country in all_language_value.keys():
            all_country_value = all_language_value.get(country)
            match_country_value = match_language_value.get(country, 0)
            accuracy = 0
            if all_country_value != 0:
                accuracy = float(
                    (Decimal(match_country_value) / Decimal(all_country_value)).quantize(Decimal('0.0001')))
            language_list.append(language)
            country_list.append(country)
            line_num_list.append(all_country_value)
            match_num_list.append(match_country_value)
            accuracy_list.append(accuracy)
            country_dict_value[country] = accuracy
        language_country_dict[language] = country_dict_value
    result = {
        "语言": language_list,
        "国家": country_list,
        "行数": line_num_list,
        "匹配行数": match_num_list,
        "准确率": accuracy_list
    }
    language_dict = dict(zip(language_list, accuracy_list))
    return {"language": language_dict, "country": language_country_dict}

def calculate_win_rate(stats):
    total_cnt = sum(stats.values())
    if not total_cnt:
        return 0
    win_cnt = stats.get("win", 0) + stats.get("tie", 0) / 2
    return win_cnt/total_cnt
