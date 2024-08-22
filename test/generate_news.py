import pandas as pd

# 扩展后的seen_data，包含多种主题
seen_data = [
    {
        "id": 1,
        "title": "2021年东京奥运会成功举办",
        "content": "2021年东京奥运会如何在疫情期间成功举办？"
    },
    {
        "id": 2,
        "title": "全球疫苗接种进展",
        "content": "各国在2021年是如何推进COVID-19疫苗接种的？"
    },
    {
        "id": 3,
        "title": "COP26气候峰会召开",
        "content": "COP26气候峰会的主要目标和成果是什么？"
    },
    {
        "id": 4,
        "title": "美国撤军阿富汗",
        "content": "美国在2021年结束阿富汗军事行动的背景是什么？"
    },
    {
        "id": 5,
        "title": "SpaceX成功发射民间航天任务",
        "content": "SpaceX在2021年进行民间航天任务的重大意义是什么？"
    },
    {
        "id": 6,
        "title": "全球供应链危机",
        "content": "2021年全球供应链危机的原因和影响是什么？"
    },
    {
        "id": 7,
        "title": "气候行动承诺",
        "content": "各国在2021年对气候行动做出了哪些新的承诺？"
    },
    {
        "id": 8,
        "title": "特斯拉发布全电动卡车",
        "content": "特斯拉发布全电动卡车Cybertruck有什么创新之处？"
    },
]



# 扩展后的unseen_data，包含错误或有误导性的内容
unseen_data = [
    {
        "id": 1,
        "title": "全球气候大会召开",
        "content": "2023年全球气候大会的主要议题是什么？"
    },
    {
        "id": 2,
        "title": "新冠疫情后经济复苏",
        "content": "各国在2023年是如何促进经济复苏的？"
    },
    {
        "id": 3,
        "title": "人工智能技术的突破",
        "content": "2023年人工智能技术在哪些领域取得了突破？"
    },
    {
        "id": 4,
        "title": "NASA新探索计划",
        "content": "NASA在2023年启动的月球和火星探索计划包括哪些内容？"
    },
    {
        "id": 5,
        "title": "全球粮食危机加剧",
        "content": "2023年全球粮食危机加剧的原因是什么？"
    },
    {
        "id": 6,
        "title": "可再生能源快速发展",
        "content": "2023年可再生能源的发展趋势和投资情况如何？"
    },
    {
        "id": 7,
        "title": "新一代电动车推出",
        "content": "2023年推出的新一代电动车有哪些特点？"
    },
    {
        "id": 8,
        "title": "国际关系的新挑战",
        "content": "2023年国际关系面临哪些新的挑战？"
    },
]



# 保存到CSV文件
seen_df = pd.DataFrame(seen_data)
unseen_df = pd.DataFrame(unseen_data)

seen_df.to_csv("dataset/seen_news.csv", index=False, encoding='utf-8')
unseen_df.to_csv("dataset/unseen_news.csv", index=False, encoding='utf-8')
