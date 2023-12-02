import os
import openai
import pandas as pd
import datetime
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine

print("Start time:", datetime.datetime.now())

# Their API key
# place key
openai.Model.list()

def call_chat_resume(text):
    print("on api call")
    completion = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      messages=[
            {
                "role": "system",
                "content": "Hello, give me the tags for this resume, make sure to also tag the years of experince and remove new lines, im going to feed it in as a string"
            },
            {
                "role": "system",
                "content": text
            }
        ]
    )
    return completion.choices[0].message

def call_chat_job(text):
    print("on api call")
    completion = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      messages=[
            {
                "role": "system",
                "content": "Hello, give me the tags for this job description, make sure to also tag the years of experince and remove new lines, im going to feed it in as a string"
            },
            {
                "role": "system",
                "content": text
            }
        ]
    )
    return completion.choices[0].message

print("Pre chat Resume tagging time:", datetime.datetime.now())
df = pd.read_csv("samp_resumes.csv")
df.drop(df.columns[0], axis=1, inplace=True)
resumes = []
resume_tracker = []
wordcloud_resume = []
for i in range(len(df)):
    try:
        resume = df.iloc[i]['col2']
        obj = call_chat_resume(resume)
        resumes.append(obj)
        resume_tracker.append(df.iloc[i]['col1'])
        wordcloud_resume.append(obj)
    except Exception as e:
        print(e)
        #print(resume)
print("Post chat resume tagging time:", datetime.datetime.now())
print(resumes)
print("Pre resume embeddings time:", datetime.datetime.now())
resume_embeddings = []
for resume in resumes:
    resume["content"] = resume["content"].replace("Tags:", "")
    resume["content"] = resume["content"].replace("tags:", "")
    response = openai.Embedding.create(
    input=resume["content"],
    model="text-embedding-ada-002"
    )
    embeddings = response['data'][0]['embedding']
    resume_embeddings.append(embeddings)
print("Post resume embeddings time:", datetime.datetime.now())

print("Pre chat Job tagging time:", datetime.datetime.now())
jobs = []
df = pd.read_csv("three_jobs.csv")
wordcloud_jobs = []
for job in df[:len(df)]["0"]:
    try:
        obj = call_chat_job(job)
        jobs.append(obj)
        wordcloud_jobs.append(obj)
    except Exception as e:
        print(e)
        print(job)
print("Post chat tagging time:", datetime.datetime.now())
#print(jobs)
print("Pre Job embeddings time:", datetime.datetime.now())
jobs_embeddings = []
for job in jobs:
    job["content"] = job["content"].replace("Tags:", "")
    job["content"] = job["content"].replace("tags:", "")
    response = openai.Embedding.create(
    input=job["content"],
    model="text-embedding-ada-002"
    )
    embeddings = response['data'][0]['embedding']
    jobs_embeddings.append(embeddings)
print("Post Job embeddings time:", datetime.datetime.now())

job_scores = []
for job in jobs_embeddings:
    scores_for_job = []
    counter = 0
    for resume in resume_embeddings:
        cosine_similarity = cosine(job, resume)
        scores_for_job.append(cosine_similarity)
        counter = counter + 1
    job_scores.append(scores_for_job)

#print(job_scores)

df = pd.DataFrame(job_scores)

column_name_mapping = {df.columns[i]: resume_tracker[i] for i in range(len(df.columns))}

df = df.rename(columns=column_name_mapping)

df.to_csv('scores.csv')
print("End time:", datetime.datetime.now())

resume_cloud = []
for resume in wordcloud_resume:
    resume_cloud.append(resume["content"])

resume_tags_combined = " ".join(resume_cloud)
resume_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(resume_tags_combined)
# Display Word Cloud for Resume Tags
plt.figure(figsize=(8, 4))
plt.imshow(resume_wordcloud, interpolation='bilinear')
plt.title('Resume Tags', loc='left', fontweight='bold', y=1.09)
plt.axis('off')
plt.show()

job_cloud = []
for job in wordcloud_jobs:
    job_cloud.append(resume["content"])

job_tags_combined = " ".join(job_cloud)
job_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(job_tags_combined)
# Display Word Cloud for Resume Tags
plt.figure(figsize=(8, 4))
plt.imshow(job_wordcloud, interpolation='bilinear')
plt.title('Job Tags', loc='left', fontweight='bold', y=1.09)
plt.axis('off')
plt.show()
