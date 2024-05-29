from openai import OpenAI
import threading
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="token-abc123",
)

def query():
    completion = client.completions.create(
    model="facebook/opt-125m",
    prompt = "London is the",
    max_tokens=200,
    n=1
    )
    print(completion)

threads = []
for i in range(1):
    thread = threading.Thread(target=query)
    threads.append(thread)
    thread.start()

for thread in threads:
    thread.join()


print("Completed")
#print(completion)