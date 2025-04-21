import os
import pandas as pd

from openai import OpenAI
from dotenv import load_dotenv


# Load OpenAI API key
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def load_data(filepath):
    try:
        df = pd.read_csv(filepath)

        # Basic preprocessing
        df = df.dropna()
        numeric_cols = ['Earnings_USD', 'Hourly_Rate', 'Job_Success_Rate',
                        'Client_Rating', 'Job_Duration_Days', 'Rehire_Rate',
                        'Marketing_Spend']
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
        df = df.dropna(subset=numeric_cols)

        # Calculate aggregate statistics
        stats = {
            'avg_earnings': df['Earnings_USD'].mean(),
            'crypto_earnings': df[df['Payment_Method'] == 'Crypto']['Earnings_USD'].mean(),
            'non_crypto_earnings': df[df['Payment_Method'] != 'Crypto']['Earnings_USD'].mean(),
            'regional_earnings': df.groupby('Client_Region')['Earnings_USD'].mean().to_dict(),
            'experience_stats': df.groupby('Experience_Level')['Earnings_USD'].mean().to_dict(),
            'category_stats': df.groupby('Job_Category')['Earnings_USD'].mean().to_dict(),
            'platform_stats': df.groupby('Platform')['Job_Success_Rate'].mean().to_dict(),
            'expert_projects': df[df['Experience_Level'] == 'Expert']['Job_Completed'].describe().to_dict(),
            'rehire_by_exp': df.groupby('Experience_Level')['Rehire_Rate'].mean().to_dict(),
            'rating_vs_earnings': df[['Client_Rating', 'Earnings_USD']].corr().iloc[0, 1],
            'duration_vs_rating': df[['Job_Duration_Days', 'Client_Rating']].corr().iloc[0, 1]
        }

        return df, stats

    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None


def generate_answer(query, stats, df_sample):
    context = f"""
    Freelancer Data Statistics:
    - General:
      * Average earnings: ${stats['avg_earnings']:.2f}
      * Crypto vs non-crypto earnings: ${stats['crypto_earnings']:.2f} vs ${stats['non_crypto_earnings']:.2f}
      * Rating vs earnings correlation: {stats['rating_vs_earnings']:.2f}
      * Job duration vs rating correlation: {stats['duration_vs_rating']:.2f}

    - By Region:
      {', '.join(f"{k}: ${v:.2f}" for k, v in stats['regional_earnings'].items())}

    - By Experience Level:
      {', '.join(f"{k}: ${v:.2f}" for k, v in stats['experience_stats'].items())}
      Rehire rates: {', '.join(f"{k}: {v:.1%}" for k, v in stats['rehire_by_exp'].items())}

    - By Job Category (top 5):
      {', '.join(f"{k}: ${v:.2f}" for k, v in sorted(stats['category_stats'].items(), key=lambda x: -x[1])[:5])}

    - Platform Statistics:
      Success rates: {', '.join(f"{k}: {v:.1%}" for k, v in stats['platform_stats'].items())}

    - Expert Freelancers:
      Projects completed: min {stats['expert_projects']['min']}, max {stats['expert_projects']['max']}, mean {stats['expert_projects']['mean']:.1f}

    Sample Data (first 3 rows):
    {df_sample.head(3).to_string()}
    """

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are a data analyst assistant. Provide concise, accurate answers based on the context."
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
                }
            ],
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating answer: {str(e)}"


def main():
    data_path = "freelancer_data.csv"
    df, stats = load_data(data_path)

    if df is None:
        print("Failed to load data. Exiting.")
        return

    print("System for analyzing freelancer earnings")
    print(f"Loaded data with {len(df)} records")
    print("\nExample questions you can ask:")
    print("- Насколько выше доход у фрилансеров, принимающих оплату в криптовалюте, по сравнению с другими способами оплаты?")
    print("- Как распределяется доход фрилансеров в зависимости от региона проживания?")
    print("- Какой процент фрилансеров, считающих себя экспертами, выполнил менее 100 проектов?")
    print("- Сравнение доходов по уровням опыта")
    print("- Рейтинг платформ по успешности проектов")
    print("- Корреляция между оценкой клиента и доходом")
    print("- Влияние метода оплаты на средний заработок")
    print("- Топ-5 самых прибыльных категорий работ\n")

    while True:
        query = input("Your question: ").strip()
        if query.lower() == 'exit':
            break
        if not query:
            continue

        answer = generate_answer(query, stats, df)
        print("\nAnswer:", answer, "\n")


if __name__ == "__main__":
    main()
