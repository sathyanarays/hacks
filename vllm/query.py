from openai import OpenAI
import threading
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="token-abc123",
)

def query():
    client.completions.create(
    model="facebook/opt-125m",
    prompt = "Sachin Tendulkar is",
    max_tokens=2040,
    n=1
    )

threads = []
for i in range(1000):
    thread = threading.Thread(target=query)
    threads.append(thread)
    thread.start()

for thread in threads:
    thread.join()


print("Completed")
#print(completion)