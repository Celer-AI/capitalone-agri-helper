Gemini API quickstart
This quickstart shows you how to install our libraries and make your first Gemini API request.

Before you begin

You need a Gemini API key. If you don't already have one, you can get it for free in Google AI Studio.

Install the Google GenAI SDK

Python
JavaScript
Go
Java
Apps Script
Using Python 3.9+, install the google-genai package using the following pip command:


pip install -q -U google-genai
Make your first request

Here is an example that uses the generateContent method to send a request to the Gemini API using the Gemini 2.5 Flash model.

If you set your API key as the environment variable GEMINI_API_KEY, it will be picked up automatically by the client when using the Gemini API libraries. Otherwise you will need to pass your API key as an argument when initializing the client.

Note that all code samples in the Gemini API docs assume that you have set the environment variable GEMINI_API_KEY.

Python
JavaScript
Go
Java
Apps Script
REST

from google import genai

# The client gets the API key from the environment variable `GEMINI_API_KEY`.
client = genai.Client()

response = client.models.generate_content(
    model="gemini-2.5-flash", contents="Explain how AI works in a few words"
)
print(response.text)
"Thinking" is on by default on many of our code samples

Many code samples on this site use the Gemini 2.5 Flash model, which has the "thinking" feature enabled by default to enhance response quality. You should be aware that this may increase response time and token usage. If you prioritize speed or wish to minimize costs, you can disable this feature by setting the thinking budget to zero, as shown in the examples below. For more details, see the thinking guide.

Note: Thinking is only available on Gemini 2.5 series models and can't be disabled on Gemini 2.5 Pro.
Python
JavaScript
Go
REST
Apps Script

from google import genai
from google.genai import types

client = genai.Client()

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Explain how AI works in a few words",
    config=types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_budget=0) # Disables thinking
    ),
)
print(response.text)
Embeddings
The Gemini API offers text embedding models to generate embeddings for words, phrases, sentences, and code. These foundational embeddings power advanced NLP tasks such as semantic search, classification, and clustering, providing more accurate, context-aware results than keyword-based approaches.

Building Retrieval Augmented Generation (RAG) systems is a common use case for embeddings. Embeddings play a key role in significantly enhancing model outputs with improved factual accuracy, coherence, and contextual richness. They efficiently retrieve relevant information from knowledge bases, represented by embeddings, which are then passed as additional context in the input prompt to language models, guiding it to generate more informed and accurate responses.

To learn more about the available embedding model variants, see the Model versions section. For enterprise-grade applications and high-volume workloads, we suggest using embedding models on Vertex AI.

Generating embeddings

Use the embedContent method to generate text embeddings:

Python
JavaScript
Go
REST

from google import genai

client = genai.Client()

result = client.models.embed_content(
        model="gemini-embedding-001",
        contents="What is the meaning of life?")

print(result.embeddings)
You can also generate embeddings for multiple chunks at once by passing them in as a list of strings.

Python
JavaScript
Go
REST

from google import genai

client = genai.Client()

result = client.models.embed_content(
        model="gemini-embedding-001",
        contents= [
            "What is the meaning of life?",
            "What is the purpose of existence?",
            "How do I bake a cake?"
        ])

for embedding in result.embeddings:
    print(embedding)
Specify task type to improve performance

You can use embeddings for a wide range of tasks from classification to document search. Specifying the right task type helps optimize the embeddings for the intended relationships, maximizing accuracy and efficiency. For a complete list of supported task types, see the Supported task types table.

The following example shows how you can use SEMANTIC_SIMILARITY to check how similar in meaning strings of texts are.

Note: Cosine similarity is a good distance metric because it focuses on direction rather than magnitude, which more accurately reflects conceptual closeness. Values range from -1 (opposite) to 1 (greatest similarity).
Python
JavaScript
Go
REST

from google import genai
from google.genai import types
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

client = genai.Client()

texts = [
    "What is the meaning of life?",
    "What is the purpose of existence?",
    "How do I bake a cake?"]

result = [
    np.array(e.values) for e in client.models.embed_content(
        model="gemini-embedding-001",
        contents=texts,
        config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY")).embeddings
]

# Calculate cosine similarity. Higher scores = greater semantic similarity.

embeddings_matrix = np.array(result)
similarity_matrix = cosine_similarity(embeddings_matrix)

for i, text1 in enumerate(texts):
    for j in range(i + 1, len(texts)):
        text2 = texts[j]
        similarity = similarity_matrix[i, j]
        print(f"Similarity between '{text1}' and '{text2}': {similarity:.4f}")
The following shows an example output from this code snippet:


Similarity between 'What is the meaning of life?' and 'What is the purpose of existence?': 0.9481

Similarity between 'What is the meaning of life?' and 'How do I bake a cake?': 0.7471

Similarity between 'What is the purpose of existence?' and 'How do I bake a cake?': 0.7371
Supported task types

Task type	Description	Examples
SEMANTIC_SIMILARITY	Embeddings optimized to assess text similarity.	Recommendation systems, duplicate detection
CLASSIFICATION	Embeddings optimized to classify texts according to preset labels.	Sentiment analysis, spam detection
CLUSTERING	Embeddings optimized to cluster texts based on their similarities.	Document organization, market research, anomaly detection
RETRIEVAL_DOCUMENT	Embeddings optimized for document search.	Indexing articles, books, or web pages for search.
RETRIEVAL_QUERY	Embeddings optimized for general search queries. Use RETRIEVAL_QUERY for queries; RETRIEVAL_DOCUMENT for documents to be retrieved.	Custom search
CODE_RETRIEVAL_QUERY	Embeddings optimized for retrieval of code blocks based on natural language queries. Use CODE_RETRIEVAL_QUERY for queries; RETRIEVAL_DOCUMENT for code blocks to be retrieved.	Code suggestions and search
QUESTION_ANSWERING	Embeddings for questions in a question-answering system, optimized for finding documents that answer the question. Use QUESTION_ANSWERING for questions; RETRIEVAL_DOCUMENT for documents to be retrieved.	Chatbox
FACT_VERIFICATION	Embeddings for statements that need to be verified, optimized for retrieving documents that contain evidence supporting or refuting the statement. Use FACT_VERIFICATION for the target text; RETRIEVAL_DOCUMENT for documents to be retrieved	Automated fact-checking systems
Controlling embedding size

The Gemini embedding model, gemini-embedding-001, is trained using the Matryoshka Representation Learning (MRL) technique which teaches a model to learn high-dimensional embeddings that have initial segments (or prefixes) which are also useful, simpler versions of the same data.

Use the output_dimensionality parameter to control the size of the output embedding vector. Selecting a smaller output dimensionality can save storage space and increase computational efficiency for downstream applications, while sacrificing little in terms of quality. By default, it outputs a 3072-dimensional embedding, but you can truncate it to a smaller size without losing quality to save storage space. We recommend using 768, 1536, or 3072 output dimensions.

Python
JavaScript
Go
REST

from google import genai
from google.genai import types

client = genai.Client()

result = client.models.embed_content(
    model="gemini-embedding-001",
    contents="What is the meaning of life?",
    config=types.EmbedContentConfig(output_dimensionality=768)
)

[embedding_obj] = result.embeddings
embedding_length = len(embedding_obj.values)

print(f"Length of embedding: {embedding_length}")
Example output from the code snippet:


Length of embedding: 768
Ensuring quality for smaller dimensions

The 3072 dimension embedding is normalized. Normalized embeddings produce more accurate semantic similarity by comparing vector direction, not magnitude. For other dimensions, including 768 and 1536, you need to normalize the embeddings as follows:

Python

import numpy as np
from numpy.linalg import norm

embedding_values_np = np.array(embedding_obj.values)
normed_embedding = embedding_values_np / np.linalg.norm(embedding_values_np)

print(f"Normed embedding length: {len(normed_embedding)}")
print(f"Norm of normed embedding: {np.linalg.norm(normed_embedding):.6f}") # Should be very close to 1
Example output from this code snippet:


Normed embedding length: 768
Norm of normed embedding: 1.000000
The following table shows the MTEB scores, a commonly used benchmark for embeddings, for different dimensions. Notably, the result shows that performance is not strictly tied to the size of the embedding dimension, with lower dimensions achieving scores comparable to their higher dimension counterparts.

MRL Dimension	MTEB Score
2048	68.16
1536	68.17
768	67.99
512	67.55
256	66.19
128	63.31
Use cases

Text embeddings are crucial for a variety of common AI use cases, such as:

Retrieval-Augmented Generation (RAG): Embeddings enhance the quality of generated text by retrieving and incorporating relevant information into the context of a model.
Information retrieval: Search for the most semantically similar text or documents given a piece of input text.

Document search tutorialtask
Search reranking: Prioritize the most relevant items by semantically scoring initial results against the query.

Search reranking tutorialtask
Anomaly detection: Comparing groups of embeddings can help identify hidden trends or outliers.

Anomaly detection tutorialbubble_chart
Classification: Automatically categorize text based on its content, such as sentiment analysis or spam detection

Classification tutorialtoken
Clustering: Effectively grasp complex relationships by creating clusters and visualizations of your embeddings.

Clustering visualization tutorialbubble_chart
Text generation
The Gemini API can generate text output from various inputs, including text, images, video, and audio, leveraging Gemini models.

Here's a basic example that takes a single text input:

Python
JavaScript
Go
REST
Apps Script

from google import genai

client = genai.Client()

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="How does AI work?"
)
print(response.text)
Thinking with Gemini 2.5

2.5 Flash and Pro models have "thinking" enabled by default to enhance quality, which may take longer to run and increase token usage.

When using 2.5 Flash, you can disable thinking by setting the thinking budget to zero.

For more details, see the thinking guide.

Python
JavaScript
Go
REST
Apps Script

from google import genai
from google.genai import types

client = genai.Client()

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="How does AI work?",
    config=types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_budget=0) # Disables thinking
    ),
)
print(response.text)
System instructions and other configurations

You can guide the behavior of Gemini models with system instructions. To do so, pass a GenerateContentConfig object.

Python
JavaScript
Go
REST
Apps Script

from google import genai
from google.genai import types

client = genai.Client()

response = client.models.generate_content(
    model="gemini-2.5-flash",
    config=types.GenerateContentConfig(
        system_instruction="You are a cat. Your name is Neko."),
    contents="Hello there"
)

print(response.text)
The GenerateContentConfig object also lets you override default generation parameters, such as temperature.

Python
JavaScript
Go
REST
Apps Script

from google import genai
from google.genai import types

client = genai.Client()

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=["Explain how AI works"],
    config=types.GenerateContentConfig(
        temperature=0.1
    )
)
print(response.text)
Refer to the GenerateContentConfig in our API reference for a complete list of configurable parameters and their descriptions.

Multimodal inputs

The Gemini API supports multimodal inputs, allowing you to combine text with media files. The following example demonstrates providing an image:

Python
JavaScript
Go
REST
Apps Script

from PIL import Image
from google import genai

client = genai.Client()

image = Image.open("/path/to/organ.png")
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=[image, "Tell me about this instrument"]
)
print(response.text)
For alternative methods of providing images and more advanced image processing, see our image understanding guide. The API also supports document, video, and audio inputs and understanding.

Streaming responses

By default, the model returns a response only after the entire generation process is complete.

For more fluid interactions, use streaming to receive GenerateContentResponse instances incrementally as they're generated.

Python
JavaScript
Go
REST
Apps Script

from google import genai

client = genai.Client()

response = client.models.generate_content_stream(
    model="gemini-2.5-flash",
    contents=["Explain how AI works"]
)
for chunk in response:
    print(chunk.text, end="")
Multi-turn conversations (Chat)

Our SDKs provide functionality to collect multiple rounds of prompts and responses into a chat, giving you an easy way to keep track of the conversation history.

Note: Chat functionality is only implemented as part of the SDKs. Behind the scenes, it still uses the generateContent API. For multi-turn conversations, the full conversation history is sent to the model with each follow-up turn.
Python
JavaScript
Go
REST
Apps Script

from google import genai

client = genai.Client()
chat = client.chats.create(model="gemini-2.5-flash")

response = chat.send_message("I have 2 dogs in my house.")
print(response.text)

response = chat.send_message("How many paws are in my house?")
print(response.text)

for message in chat.get_history():
    print(f'role - {message.role}',end=": ")
    print(message.parts[0].text)
Streaming can also be used for multi-turn conversations.

Python
JavaScript
Go
REST
Apps Script

from google import genai

client = genai.Client()
chat = client.chats.create(model="gemini-2.5-flash")

response = chat.send_message_stream("I have 2 dogs in my house.")
for chunk in response:
    print(chunk.text, end="")

response = chat.send_message_stream("How many paws are in my house?")
for chunk in response:
    print(chunk.text, end="")

for message in chat.get_history():
    print(f'role - {message.role}', end=": ")
    print(message.parts[0].text)
Supported models

All models in the Gemini family support text generation. To learn more about the models and their capabilities, visit the Models page.

Best practices

Prompting tips

For basic text generation, a zero-shot prompt often suffices without needing examples, system instructions or specific formatting.

For more tailored outputs:

Use System instructions to guide the model.
Provide few example inputs and outputs to guide the model. This is often referred to as few-shot prompting.
Consult our prompt engineering guide for more tips.

Structured output

In some cases, you may need structured output, such as JSON. Refer to our structured output guide to learn how.

Speech generation (text-to-speech)
The Gemini API can transform text input into single speaker or multi-speaker audio using native text-to-speech (TTS) generation capabilities. Text-to-speech (TTS) generation is controllable, meaning you can use natural language to structure interactions and guide the style, accent, pace, and tone of the audio.

The TTS capability differs from speech generation provided through the Live API, which is designed for interactive, unstructured audio, and multimodal inputs and outputs. While the Live API excels in dynamic conversational contexts, TTS through the Gemini API is tailored for scenarios that require exact text recitation with fine-grained control over style and sound, such as podcast or audiobook generation.

This guide shows you how to generate single-speaker and multi-speaker audio from text.

Preview: Native text-to-speech (TTS) is in Preview.
Before you begin

Ensure you use a Gemini 2.5 model variant with native text-to-speech (TTS) capabilities, as listed in the Supported models section. For optimal results, consider which model best fits your specific use case.

You may find it useful to test the Gemini 2.5 TTS models in AI Studio before you start building.

Note: TTS models accept text-only inputs and produce audio-only outputs. For a complete list of restrictions specific to TTS models, review the Limitations section.
Single-speaker text-to-speech

To convert text to single-speaker audio, set the response modality to "audio", and pass a SpeechConfig object with VoiceConfig set. You'll need to choose a voice name from the prebuilt output voices.

This example saves the output audio from the model in a wave file:

Python
JavaScript
REST

from google import genai
from google.genai import types
import wave

# Set up the wave file to save the output:
def wave_file(filename, pcm, channels=1, rate=24000, sample_width=2):
   with wave.open(filename, "wb") as wf:
      wf.setnchannels(channels)
      wf.setsampwidth(sample_width)
      wf.setframerate(rate)
      wf.writeframes(pcm)

client = genai.Client()

response = client.models.generate_content(
   model="gemini-2.5-flash-preview-tts",
   contents="Say cheerfully: Have a wonderful day!",
   config=types.GenerateContentConfig(
      response_modalities=["AUDIO"],
      speech_config=types.SpeechConfig(
         voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(
               voice_name='Kore',
            )
         )
      ),
   )
)

data = response.candidates[0].content.parts[0].inline_data.data

file_name='out.wav'
wave_file(file_name, data) # Saves the file to current directory
For more code samples, refer to the "TTS - Get Started" file in the cookbooks repository:

View on GitHub
Multi-speaker text-to-speech

For multi-speaker audio, you'll need a MultiSpeakerVoiceConfig object with each speaker (up to 2) configured as a SpeakerVoiceConfig. You'll need to define each speaker with the same names used in the prompt:

Python
JavaScript
REST

from google import genai
from google.genai import types
import wave

# Set up the wave file to save the output:
def wave_file(filename, pcm, channels=1, rate=24000, sample_width=2):
   with wave.open(filename, "wb") as wf:
      wf.setnchannels(channels)
      wf.setsampwidth(sample_width)
      wf.setframerate(rate)
      wf.writeframes(pcm)

client = genai.Client()

prompt = """TTS the following conversation between Joe and Jane:
         Joe: How's it going today Jane?
         Jane: Not too bad, how about you?"""

response = client.models.generate_content(
   model="gemini-2.5-flash-preview-tts",
   contents=prompt,
   config=types.GenerateContentConfig(
      response_modalities=["AUDIO"],
      speech_config=types.SpeechConfig(
         multi_speaker_voice_config=types.MultiSpeakerVoiceConfig(
            speaker_voice_configs=[
               types.SpeakerVoiceConfig(
                  speaker='Joe',
                  voice_config=types.VoiceConfig(
                     prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name='Kore',
                     )
                  )
               ),
               types.SpeakerVoiceConfig(
                  speaker='Jane',
                  voice_config=types.VoiceConfig(
                     prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name='Puck',
                     )
                  )
               ),
            ]
         )
      )
   )
)

data = response.candidates[0].content.parts[0].inline_data.data

file_name='out.wav'
wave_file(file_name, data) # Saves the file to current directory
Controlling speech style with prompts

You can control style, tone, accent, and pace using natural language prompts for both single- and multi-speaker TTS. For example, in a single-speaker prompt, you can say:


Say in an spooky whisper:
"By the pricking of my thumbs...
Something wicked this way comes"
In a multi-speaker prompt, provide the model with each speaker's name and corresponding transcript. You can also provide guidance for each speaker individually:


Make Speaker1 sound tired and bored, and Speaker2 sound excited and happy:

Speaker1: So... what's on the agenda today?
Speaker2: You're never going to guess!
Try using a voice option that corresponds to the style or emotion you want to convey, to emphasize it even more. In the previous prompt, for example, Enceladus's breathiness might emphasize "tired" and "bored", while Puck's upbeat tone could complement "excited" and "happy".

Generating a prompt to convert to audio

The TTS models only output audio, but you can use other models to generate a transcript first, then pass that transcript to the TTS model to read aloud.

Python
JavaScript

from google import genai
from google.genai import types

client = genai.Client()

transcript = client.models.generate_content(
   model="gemini-2.0-flash",
   contents="""Generate a short transcript around 100 words that reads
            like it was clipped from a podcast by excited herpetologists.
            The hosts names are Dr. Anya and Liam.""").text

response = client.models.generate_content(
   model="gemini-2.5-flash-preview-tts",
   contents=transcript,
   config=types.GenerateContentConfig(
      response_modalities=["AUDIO"],
      speech_config=types.SpeechConfig(
         multi_speaker_voice_config=types.MultiSpeakerVoiceConfig(
            speaker_voice_configs=[
               types.SpeakerVoiceConfig(
                  speaker='Dr. Anya',
                  voice_config=types.VoiceConfig(
                     prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name='Kore',
                     )
                  )
               ),
               types.SpeakerVoiceConfig(
                  speaker='Liam',
                  voice_config=types.VoiceConfig(
                     prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name='Puck',
                     )
                  )
               ),
            ]
         )
      )
   )
)

# ...Code to stream or save the output
Voice options

TTS models support the following 30 voice options in the voice_name field:

Zephyr -- Bright	Puck -- Upbeat	Charon -- Informative
Kore -- Firm	Fenrir -- Excitable	Leda -- Youthful
Orus -- Firm	Aoede -- Breezy	Callirrhoe -- Easy-going
Autonoe -- Bright	Enceladus -- Breathy	Iapetus -- Clear
Umbriel -- Easy-going	Algieba -- Smooth	Despina -- Smooth
Erinome -- Clear	Algenib -- Gravelly	Rasalgethi -- Informative
Laomedeia -- Upbeat	Achernar -- Soft	Alnilam -- Firm
Schedar -- Even	Gacrux -- Mature	Pulcherrima -- Forward
Achird -- Friendly	Zubenelgenubi -- Casual	Vindemiatrix -- Gentle
Sadachbia -- Lively	Sadaltager -- Knowledgeable	Sulafat -- Warm
You can hear all the voice options in AI Studio.

Supported languages

The TTS models detect the input language automatically. They support the following 24 languages:

Language	BCP-47 Code	Language	BCP-47 Code
Arabic (Egyptian)	ar-EG	German (Germany)	de-DE
English (US)	en-US	Spanish (US)	es-US
French (France)	fr-FR	Hindi (India)	hi-IN
Indonesian (Indonesia)	id-ID	Italian (Italy)	it-IT
Japanese (Japan)	ja-JP	Korean (Korea)	ko-KR
Portuguese (Brazil)	pt-BR	Russian (Russia)	ru-RU
Dutch (Netherlands)	nl-NL	Polish (Poland)	pl-PL
Thai (Thailand)	th-TH	Turkish (Turkey)	tr-TR
Vietnamese (Vietnam)	vi-VN	Romanian (Romania)	ro-RO
Ukrainian (Ukraine)	uk-UA	Bengali (Bangladesh)	bn-BD
English (India)	en-IN & hi-IN bundle	Marathi (India)	mr-IN
Tamil (India)	ta-IN	Telugu (India)	te-IN
Supported models

Model	Single speaker	Multispeaker
Gemini 2.5 Flash Preview TTS	✔️	✔️
Gemini 2.5 Pro Preview TTS	✔️	✔️
Gemini thinking
The Gemini 2.5 series models use an internal "thinking process" that significantly improves their reasoning and multi-step planning abilities, making them highly effective for complex tasks such as coding, advanced mathematics, and data analysis.

This guide shows you how to work with Gemini's thinking capabilities using the Gemini API.

Before you begin

Ensure you use a supported 2.5 series model for thinking. You might find it beneficial to explore these models in AI Studio before diving into the API:

Try Gemini 2.5 Flash in AI Studio
Try Gemini 2.5 Pro in AI Studio
Try Gemini 2.5 Flash-Lite in AI Studio
Generating content with thinking

Initiating a request with a thinking model is similar to any other content generation request. The key difference lies in specifying one of the models with thinking support in the model field, as demonstrated in the following text generation example:

Python
JavaScript
Go
REST

from google import genai

client = genai.Client()
prompt = "Explain the concept of Occam's Razor and provide a simple, everyday example."
response = client.models.generate_content(
    model="gemini-2.5-pro",
    contents=prompt
)

print(response.text)
Thinking budgets

The thinkingBudget parameter guides the model on the number of thinking tokens to use when generating a response. A higher token count generally allows for more detailed reasoning, which can be beneficial for tackling more complex tasks. If latency is more important, use a lower budget or disable thinking by setting thinkingBudget to 0. Setting the thinkingBudget to -1 turns on dynamic thinking, meaning the model will adjust the budget based on the complexity of the request.

The thinkingBudget is only supported in Gemini 2.5 Flash, 2.5 Pro, and 2.5 Flash-Lite. Depending on the prompt, the model might overflow or underflow the token budget.

The following are thinkingBudget configuration details for each model type.

Model	Default setting
(Thinking budget is not set)	Range	Disable thinking	Turn on dynamic thinking
2.5 Pro	Dynamic thinking: Model decides when and how much to think	128 to 32768	N/A: Cannot disable thinking	thinkingBudget = -1
2.5 Flash	Dynamic thinking: Model decides when and how much to think	0 to 24576	thinkingBudget = 0	thinkingBudget = -1
2.5 Flash Lite	Model does not think	512 to 24576	thinkingBudget = 0	thinkingBudget = -1
Python
JavaScript
Go
REST

from google import genai
from google.genai import types

client = genai.Client()

response = client.models.generate_content(
    model="gemini-2.5-pro",
    contents="Provide a list of 3 famous physicists and their key contributions",
    config=types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_budget=1024)
        # Turn off thinking:
        # thinking_config=types.ThinkingConfig(thinking_budget=0)
        # Turn on dynamic thinking:
        # thinking_config=types.ThinkingConfig(thinking_budget=-1)
    ),
)

print(response.text)
Thought summaries

Thought summaries are synthesized versions of the model's raw thoughts and offer insights into the model's internal reasoning process. Note that thinking budgets apply to the model's raw thoughts and not to thought summaries.

You can enable thought summaries by setting includeThoughts to true in your request configuration. You can then access the summary by iterating through the response parameter's parts, and checking the thought boolean.

Here's an example demonstrating how to enable and retrieve thought summaries without streaming, which returns a single, final thought summary with the response:

Python
JavaScript
Go

from google import genai
from google.genai import types

client = genai.Client()
prompt = "What is the sum of the first 50 prime numbers?"
response = client.models.generate_content(
  model="gemini-2.5-pro",
  contents=prompt,
  config=types.GenerateContentConfig(
    thinking_config=types.ThinkingConfig(
      include_thoughts=True
    )
  )
)

for part in response.candidates[0].content.parts:
  if not part.text:
    continue
  if part.thought:
    print("Thought summary:")
    print(part.text)
    print()
  else:
    print("Answer:")
    print(part.text)
    print()
And here is an example using thinking with streaming, which returns rolling, incremental summaries during generation:

Python
JavaScript
Go

from google import genai
from google.genai import types

client = genai.Client()

prompt = """
Alice, Bob, and Carol each live in a different house on the same street: red, green, and blue.
The person who lives in the red house owns a cat.
Bob does not live in the green house.
Carol owns a dog.
The green house is to the left of the red house.
Alice does not own a cat.
Who lives in each house, and what pet do they own?
"""

thoughts = ""
answer = ""

for chunk in client.models.generate_content_stream(
    model="gemini-2.5-pro",
    contents=prompt,
    config=types.GenerateContentConfig(
      thinking_config=types.ThinkingConfig(
        include_thoughts=True
      )
    )
):
  for part in chunk.candidates[0].content.parts:
    if not part.text:
      continue
    elif part.thought:
      if not thoughts:
        print("Thoughts summary:")
      print(part.text)
      thoughts += part.text
    else:
      if not answer:
        print("Answer:")
      print(part.text)
      answer += part.text
Thought signatures

Because standard Gemini API text and content generation calls are stateless, when using thinking in multi-turn interactions (such as chat), the model doesn't have access to thought context from previous turns.

You can maintain thought context using thought signatures, which are encrypted representations of the model's internal thought process. The model returns thought signatures in the response object when thinking and function calling are enabled. To ensure the model maintains context across multiple turns of a conversation, you must provide the thought signatures back to the model in the subsequent requests.

You will receive thought signatures when:

Thinking is enabled and thoughts are generated.
The request includes function declarations.
Note: Thought signatures are only available when you're using function calling, specifically, your request must include function declarations.
You can find an example of thinking with function calls on the Function calling page.

Other usage limitations to consider with function calling include:

Signatures are returned from the model within other parts in the response, for example function calling or text parts. Return the entire response with all parts back to the model in subsequent turns.
Don't concatenate parts with signatures together.
Don't merge one part with a signature with another part without a signature.
Document understanding
Gemini models can process documents in PDF format, using native vision to understand entire document contexts. This goes beyond simple text extraction, allowing Gemini to:

Analyze and interpret content, including text, images, diagrams, charts, and tables, even in long documents up to 1000 pages.
Extract information into structured output formats.
Summarize and answer questions based on both the visual and textual elements in a document.
Transcribe document content (e.g. to HTML), preserving layouts and formatting, for use in downstream applications.
Passing inline PDF data

You can pass inline PDF data in the request to generateContent. For PDF payloads under 20MB, you can choose between uploading base64 encoded documents or directly uploading locally stored files.

The following example shows you how to fetch a PDF from a URL and convert it to bytes for processing:

Python
JavaScript
Go
REST

from google import genai
from google.genai import types
import httpx

client = genai.Client()

doc_url = "https://discovery.ucl.ac.uk/id/eprint/10089234/1/343019_3_art_0_py4t4l_convrt.pdf"

# Retrieve and encode the PDF byte
doc_data = httpx.get(doc_url).content

prompt = "Summarize this document"
response = client.models.generate_content(
  model="gemini-2.5-flash",
  contents=[
      types.Part.from_bytes(
        data=doc_data,
        mime_type='application/pdf',
      ),
      prompt])
print(response.text)
You can also read a PDF from a local file for processing:

Python
JavaScript
Go

from google import genai
from google.genai import types
import pathlib

client = genai.Client()

# Retrieve and encode the PDF byte
filepath = pathlib.Path('file.pdf')

prompt = "Summarize this document"
response = client.models.generate_content(
  model="gemini-2.5-flash",
  contents=[
      types.Part.from_bytes(
        data=filepath.read_bytes(),
        mime_type='application/pdf',
      ),
      prompt])
print(response.text)
Uploading PDFs using the File API

You can use the File API to upload larger documents. Always use the File API when the total request size (including the files, text prompt, system instructions, etc.) is larger than 20MB.

Note: The File API lets you store up to 50MB of PDF files. Files are stored for 48 hours. You can access them in that period with your API key, but you can't download them from the API. The File API is available at no cost in all regions where the Gemini API is available.
Call media.upload to upload a file using the File API. The following code uploads a document file and then uses the file in a call to models.generateContent.

Large PDFs from URLs

Use the File API to simplify uploading and processing large PDF files from URLs:

Python
JavaScript
Go
REST

from google import genai
from google.genai import types
import io
import httpx

client = genai.Client()

long_context_pdf_path = "https://www.nasa.gov/wp-content/uploads/static/history/alsj/a17/A17_FlightPlan.pdf"

# Retrieve and upload the PDF using the File API
doc_io = io.BytesIO(httpx.get(long_context_pdf_path).content)

sample_doc = client.files.upload(
  # You can pass a path or a file-like object here
  file=doc_io,
  config=dict(
    mime_type='application/pdf')
)

prompt = "Summarize this document"

response = client.models.generate_content(
  model="gemini-2.5-flash",
  contents=[sample_doc, prompt])
print(response.text)
Large PDFs stored locally

Python
JavaScript
Go
REST

from google import genai
from google.genai import types
import pathlib
import httpx

client = genai.Client()

# Retrieve and encode the PDF byte
file_path = pathlib.Path('large_file.pdf')

# Upload the PDF using the File API
sample_file = client.files.upload(
  file=file_path,
)

prompt="Summarize this document"

response = client.models.generate_content(
  model="gemini-2.5-flash",
  contents=[sample_file, "Summarize this document"])
print(response.text)
You can verify the API successfully stored the uploaded file and get its metadata by calling files.get. Only the name (and by extension, the uri) are unique.

Python
REST

from google import genai
import pathlib

client = genai.Client()

fpath = pathlib.Path('example.txt')
fpath.write_text('hello')

file = client.files.upload(file='example.txt')

file_info = client.files.get(name=file.name)
print(file_info.model_dump_json(indent=4))
Passing multiple PDFs

The Gemini API is capable of processing multiple PDF documents (up to 1000 pages) in a single request, as long as the combined size of the documents and the text prompt stays within the model's context window.

Python
JavaScript
Go
REST

from google import genai
import io
import httpx

client = genai.Client()

doc_url_1 = "https://arxiv.org/pdf/2312.11805"
doc_url_2 = "https://arxiv.org/pdf/2403.05530"

# Retrieve and upload both PDFs using the File API
doc_data_1 = io.BytesIO(httpx.get(doc_url_1).content)
doc_data_2 = io.BytesIO(httpx.get(doc_url_2).content)

sample_pdf_1 = client.files.upload(
  file=doc_data_1,
  config=dict(mime_type='application/pdf')
)
sample_pdf_2 = client.files.upload(
  file=doc_data_2,
  config=dict(mime_type='application/pdf')
)

prompt = "What is the difference between each of the main benchmarks between these two papers? Output these in a table."

response = client.models.generate_content(
  model="gemini-2.5-flash",
  contents=[sample_pdf_1, sample_pdf_2, prompt])
print(response.text)
Technical details

Gemini supports a maximum of 1,000 document pages. Each document page is equivalent to 258 tokens.

While there are no specific limits to the number of pixels in a document besides the model's context window, larger pages are scaled down to a maximum resolution of 3072x3072 while preserving their original aspect ratio, while smaller pages are scaled up to 768x768 pixels. There is no cost reduction for pages at lower sizes, other than bandwidth, or performance improvement for pages at higher resolution.

Document types

Technically, you can pass other MIME types for document understanding, like TXT, Markdown, HTML, XML, etc. However, document vision only meaningfully understands PDFs. Other types will be extracted as pure text, and the model won't be able to interpret what we see in the rendering of those files. Any file-type specifics like charts, diagrams, HTML tags, Markdown formatting, etc., will be lost.

Audio understanding
Gemini can analyze and understand audio input, enabling use cases like the following:

Describe, summarize, or answer questions about audio content.
Provide a transcription of the audio.
Analyze specific segments of the audio.
This guide shows you how to use the Gemini API to generate a text response to audio input.

Before you begin

Before calling the Gemini API, ensure you have your SDK of choice installed, and a Gemini API key configured and ready to use.

Input audio

You can provide audio data to Gemini in the following ways:

Upload an audio file before making a request to generateContent.
Pass inline audio data with the request to generateContent.
Upload an audio file

You can use the Files API to upload an audio file. Always use the Files API when the total request size (including the files, text prompt, system instructions, etc.) is larger than 20 MB.

The following code uploads an audio file and then uses the file in a call to generateContent.

Python
JavaScript
Go
REST

from google import genai

client = genai.Client()

myfile = client.files.upload(file="path/to/sample.mp3")

response = client.models.generate_content(
    model="gemini-2.5-flash", contents=["Describe this audio clip", myfile]
)

print(response.text)
To learn more about working with media files, see Files API.

Pass audio data inline

Instead of uploading an audio file, you can pass inline audio data in the request to generateContent:

Python
JavaScript
Go

from google.genai import types

with open('path/to/small-sample.mp3', 'rb') as f:
    audio_bytes = f.read()

response = client.models.generate_content(
  model='gemini-2.5-flash',
  contents=[
    'Describe this audio clip',
    types.Part.from_bytes(
      data=audio_bytes,
      mime_type='audio/mp3',
    )
  ]
)

print(response.text)
A few things to keep in mind about inline audio data:

The maximum request size is 20 MB, which includes text prompts, system instructions, and files provided inline. If your file's size will make the total request size exceed 20 MB, then use the Files API to upload an audio file for use in the request.
If you're using an audio sample multiple times, it's more efficient to upload an audio file.
Get a transcript

To get a transcript of audio data, just ask for it in the prompt:

Python
JavaScript
Go

myfile = client.files.upload(file='path/to/sample.mp3')
prompt = 'Generate a transcript of the speech.'

response = client.models.generate_content(
  model='gemini-2.5-flash',
  contents=[prompt, myfile]
)

print(response.text)
Refer to timestamps

You can refer to specific sections of an audio file using timestamps of the form MM:SS. For example, the following prompt requests a transcript that

Starts at 2 minutes 30 seconds from the beginning of the file.
Ends at 3 minutes 29 seconds from the beginning of the file.
Python
JavaScript
Go

# Create a prompt containing timestamps.
prompt = "Provide a transcript of the speech from 02:30 to 03:29."
Count tokens

Call the countTokens method to get a count of the number of tokens in an audio file. For example:

Python
JavaScript
Go

response = client.models.count_tokens(
  model='gemini-2.5-flash',
  contents=[myfile]
)

print(response)
Supported audio formats

Gemini supports the following audio format MIME types:

WAV - audio/wav
MP3 - audio/mp3
AIFF - audio/aiff
AAC - audio/aac
OGG Vorbis - audio/ogg
FLAC - audio/flac

Function calling with the Gemini API
Function calling lets you connect models to external tools and APIs. Instead of generating text responses, the model determines when to call specific functions and provides the necessary parameters to execute real-world actions. This allows the model to act as a bridge between natural language and real-world actions and data. Function calling has 3 primary use cases:

Augment Knowledge: Access information from external sources like databases, APIs, and knowledge bases.
Extend Capabilities: Use external tools to perform computations and extend the limitations of the model, such as using a calculator or creating charts.
Take Actions: Interact with external systems using APIs, such as scheduling appointments, creating invoices, sending emails, or controlling smart home devices.
Get Weather  Schedule Meeting  Create Chart

Python
JavaScript
REST

from google import genai
from google.genai import types

# Define the function declaration for the model
schedule_meeting_function = {
    "name": "schedule_meeting",
    "description": "Schedules a meeting with specified attendees at a given time and date.",
    "parameters": {
        "type": "object",
        "properties": {
            "attendees": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of people attending the meeting.",
            },
            "date": {
                "type": "string",
                "description": "Date of the meeting (e.g., '2024-07-29')",
            },
            "time": {
                "type": "string",
                "description": "Time of the meeting (e.g., '15:00')",
            },
            "topic": {
                "type": "string",
                "description": "The subject or topic of the meeting.",
            },
        },
        "required": ["attendees", "date", "time", "topic"],
    },
}

# Configure the client and tools
client = genai.Client()
tools = types.Tool(function_declarations=[schedule_meeting_function])
config = types.GenerateContentConfig(tools=[tools])

# Send request with function declarations
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Schedule a meeting with Bob and Alice for 03/14/2025 at 10:00 AM about the Q3 planning.",
    config=config,
)

# Check for a function call
if response.candidates[0].content.parts[0].function_call:
    function_call = response.candidates[0].content.parts[0].function_call
    print(f"Function to call: {function_call.name}")
    print(f"Arguments: {function_call.args}")
    #  In a real app, you would call your function here:
    #  result = schedule_meeting(**function_call.args)
else:
    print("No function call found in the response.")
    print(response.text)
How function calling works

function calling
overview

Function calling involves a structured interaction between your application, the model, and external functions. Here's a breakdown of the process:

Define Function Declaration: Define the function declaration in your application code. Function Declarations describe the function's name, parameters, and purpose to the model.
Call LLM with function declarations: Send user prompt along with the function declaration(s) to the model. It analyzes the request and determines if a function call would be helpful. If so, it responds with a structured JSON object.
Execute Function Code (Your Responsibility): The Model does not execute the function itself. It's your application's responsibility to process the response and check for Function Call, if
Yes: Extract the name and args of the function and execute the corresponding function in your application.
No: The model has provided a direct text response to the prompt (this flow is less emphasized in the example but is a possible outcome).
Create User friendly response: If a function was executed, capture the result and send it back to the model in a subsequent turn of the conversation. It will use the result to generate a final, user-friendly response that incorporates the information from the function call.
This process can be repeated over multiple turns, allowing for complex interactions and workflows. The model also supports calling multiple functions in a single turn (parallel function calling) and in sequence (compositional function calling).

Step 1: Define a function declaration

Define a function and its declaration within your application code that allows users to set light values and make an API request. This function could call external services or APIs.

Python
JavaScript

# Define a function that the model can call to control smart lights
set_light_values_declaration = {
    "name": "set_light_values",
    "description": "Sets the brightness and color temperature of a light.",
    "parameters": {
        "type": "object",
        "properties": {
            "brightness": {
                "type": "integer",
                "description": "Light level from 0 to 100. Zero is off and 100 is full brightness",
            },
            "color_temp": {
                "type": "string",
                "enum": ["daylight", "cool", "warm"],
                "description": "Color temperature of the light fixture, which can be `daylight`, `cool` or `warm`.",
            },
        },
        "required": ["brightness", "color_temp"],
    },
}

# This is the actual function that would be called based on the model's suggestion
def set_light_values(brightness: int, color_temp: str) -> dict[str, int | str]:
    """Set the brightness and color temperature of a room light. (mock API).

    Args:
        brightness: Light level from 0 to 100. Zero is off and 100 is full brightness
        color_temp: Color temperature of the light fixture, which can be `daylight`, `cool` or `warm`.

    Returns:
        A dictionary containing the set brightness and color temperature.
    """
    return {"brightness": brightness, "colorTemperature": color_temp}
Step 2: Call the model with function declarations

Once you have defined your function declarations, you can prompt the model to use them. It analyzes the prompt and function declarations and decides whether to respond directly or to call a function. If a function is called, the response object will contain a function call suggestion.

Python
JavaScript

from google.genai import types

# Configure the client and tools
client = genai.Client()
tools = types.Tool(function_declarations=[set_light_values_declaration])
config = types.GenerateContentConfig(tools=[tools])

# Define user prompt
contents = [
    types.Content(
        role="user", parts=[types.Part(text="Turn the lights down to a romantic level")]
    )
]

# Send request with function declarations
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=contents
    config=config,
)

print(response.candidates[0].content.parts[0].function_call)
The model then returns a functionCall object in an OpenAPI compatible schema specifying how to call one or more of the declared functions in order to respond to the user's question.

Python
JavaScript

id=None args={'color_temp': 'warm', 'brightness': 25} name='set_light_values'
Step 3: Execute set_light_values function code

Extract the function call details from the model's response, parse the arguments , and execute the set_light_values function.

Python
JavaScript

# Extract tool call details, it may not be in the first part.
tool_call = response.candidates[0].content.parts[0].function_call

if tool_call.name == "set_light_values":
    result = set_light_values(**tool_call.args)
    print(f"Function execution result: {result}")
Step 4: Create user friendly response with function result and call the model again

Finally, send the result of the function execution back to the model so it can incorporate this information into its final response to the user.

Python
JavaScript

# Create a function response part
function_response_part = types.Part.from_function_response(
    name=tool_call.name,
    response={"result": result},
)

# Append function call and result of the function execution to contents
contents.append(response.candidates[0].content) # Append the content from the model's response.
contents.append(types.Content(role="user", parts=[function_response_part])) # Append the function response

final_response = client.models.generate_content(
    model="gemini-2.5-flash",
    config=config,
    contents=contents,
)

print(final_response.text)
This completes the function calling flow. The model successfully used the set_light_values function to perform the request action of the user.

Function declarations

When you implement function calling in a prompt, you create a tools object, which contains one or more function declarations. You define functions using JSON, specifically with a select subset of the OpenAPI schema format. A single function declaration can include the following parameters:

name (string): A unique name for the function (get_weather_forecast, send_email). Use descriptive names without spaces or special characters (use underscores or camelCase).
description (string): A clear and detailed explanation of the function's purpose and capabilities. This is crucial for the model to understand when to use the function. Be specific and provide examples if helpful ("Finds theaters based on location and optionally movie title which is currently playing in theaters.").
parameters (object): Defines the input parameters the function expects.
type (string): Specifies the overall data type, such as object.
properties (object): Lists individual parameters, each with:
type (string): The data type of the parameter, such as string, integer, boolean, array.
description (string): A description of the parameter's purpose and format. Provide examples and constraints ("The city and state, e.g., 'San Francisco, CA' or a zip code e.g., '95616'.").
enum (array, optional): If the parameter values are from a fixed set, use "enum" to list the allowed values instead of just describing them in the description. This improves accuracy ("enum": ["daylight", "cool", "warm"]).
required (array): An array of strings listing the parameter names that are mandatory for the function to operate.
You can also construct FunctionDeclarations from Python functions directly using types.FunctionDeclaration.from_callable(client=client, callable=your_function).

Function calling with thinking

Enabling "thinking" can improve function call performance by allowing the model to reason through a request before suggesting function calls. The Gemini API is stateless, the model's reasoning context will be lost between turns in a multi-turn conversation. To preserve this context, you can use thought signatures. A thought signature is an encrypted representation of the model's internal thought process that you pass back to the model on subsequent turns.

The standard pattern for multi-turn tool use is to append the model's complete previous response to the conversation history. The content object includes the thought_signatures automatically. If you follow this pattern No code changes are required.

Manually managing thought signatures

If you modify the conversation history manually—instead of sending the complete previous response and want to benefit from thinking you must correctly handle the thought_signature included in the model's turn.

Follow these rules to ensure the model's context is preserved:

Always send the thought_signature back to the model inside its original Part.
Don't merge a Part containing a signature with one that does not. This breaks the positional context of the thought.
Don't combine two Parts that both contain signatures, as the signature strings cannot be merged.
Inspecting Thought Signatures

While not necessary for implementation, you can inspect the response to see the thought_signature for debugging or educational purposes.

Python
JavaScript

import base64
# After receiving a response from a model with thinking enabled
# response = client.models.generate_content(...)

# The signature is attached to the response part containing the function call
part = response.candidates[0].content.parts[0]
if part.thought_signature:
  print(base64.b64encode(part.thought_signature).decode("utf-8"))
Learn more about limitations and usage of thought signatures, and about thinking models in general, on the Thinking page.

Parallel function calling

In addition to single turn function calling, you can also call multiple functions at once. Parallel function calling lets you execute multiple functions at once and is used when the functions are not dependent on each other. This is useful in scenarios like gathering data from multiple independent sources, such as retrieving customer details from different databases or checking inventory levels across various warehouses or performing multiple actions such as converting your apartment into a disco.

Python
JavaScript

power_disco_ball = {
    "name": "power_disco_ball",
    "description": "Powers the spinning disco ball.",
    "parameters": {
        "type": "object",
        "properties": {
            "power": {
                "type": "boolean",
                "description": "Whether to turn the disco ball on or off.",
            }
        },
        "required": ["power"],
    },
}

start_music = {
    "name": "start_music",
    "description": "Play some music matching the specified parameters.",
    "parameters": {
        "type": "object",
        "properties": {
            "energetic": {
                "type": "boolean",
                "description": "Whether the music is energetic or not.",
            },
            "loud": {
                "type": "boolean",
                "description": "Whether the music is loud or not.",
            },
        },
        "required": ["energetic", "loud"],
    },
}

dim_lights = {
    "name": "dim_lights",
    "description": "Dim the lights.",
    "parameters": {
        "type": "object",
        "properties": {
            "brightness": {
                "type": "number",
                "description": "The brightness of the lights, 0.0 is off, 1.0 is full.",
            }
        },
        "required": ["brightness"],
    },
}
Configure the function calling mode to allow using all of the specified tools. To learn more, you can read about configuring function calling.

Python
JavaScript

from google import genai
from google.genai import types

# Configure the client and tools
client = genai.Client()
house_tools = [
    types.Tool(function_declarations=[power_disco_ball, start_music, dim_lights])
]
config = types.GenerateContentConfig(
    tools=house_tools,
    automatic_function_calling=types.AutomaticFunctionCallingConfig(
        disable=True
    ),
    # Force the model to call 'any' function, instead of chatting.
    tool_config=types.ToolConfig(
        function_calling_config=types.FunctionCallingConfig(mode='ANY')
    ),
)

chat = client.chats.create(model="gemini-2.5-flash", config=config)
response = chat.send_message("Turn this place into a party!")

# Print out each of the function calls requested from this single call
print("Example 1: Forced function calling")
for fn in response.function_calls:
    args = ", ".join(f"{key}={val}" for key, val in fn.args.items())
    print(f"{fn.name}({args})")
Each of the printed results reflects a single function call that the model has requested. To send the results back, include the responses in the same order as they were requested.

The Python SDK supports automatic function calling, which automatically converts Python functions to declarations, handles the function call execution and response cycle for you. Following is an example for the disco use case.

Note: Automatic Function Calling is a Python SDK only feature at the moment.
Python

from google import genai
from google.genai import types

# Actual function implementations
def power_disco_ball_impl(power: bool) -> dict:
    """Powers the spinning disco ball.

    Args:
        power: Whether to turn the disco ball on or off.

    Returns:
        A status dictionary indicating the current state.
    """
    return {"status": f"Disco ball powered {'on' if power else 'off'}"}

def start_music_impl(energetic: bool, loud: bool) -> dict:
    """Play some music matching the specified parameters.

    Args:
        energetic: Whether the music is energetic or not.
        loud: Whether the music is loud or not.

    Returns:
        A dictionary containing the music settings.
    """
    music_type = "energetic" if energetic else "chill"
    volume = "loud" if loud else "quiet"
    return {"music_type": music_type, "volume": volume}

def dim_lights_impl(brightness: float) -> dict:
    """Dim the lights.

    Args:
        brightness: The brightness of the lights, 0.0 is off, 1.0 is full.

    Returns:
        A dictionary containing the new brightness setting.
    """
    return {"brightness": brightness}

# Configure the client
client = genai.Client()
config = types.GenerateContentConfig(
    tools=[power_disco_ball_impl, start_music_impl, dim_lights_impl]
)

# Make the request
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Do everything you need to this place into party!",
    config=config,
)

print("\nExample 2: Automatic function calling")
print(response.text)
# I've turned on the disco ball, started playing loud and energetic music, and dimmed the lights to 50% brightness. Let's get this party started!
Compositional function calling

Compositional or sequential function calling allows Gemini to chain multiple function calls together to fulfill a complex request. For example, to answer "Get the temperature in my current location", the Gemini API might first invoke a get_current_location() function followed by a get_weather() function that takes the location as a parameter.

The following example demonstrates how to implement compositional function calling using the Python SDK and automatic function calling.

Python
JavaScript
This example uses the automatic function calling feature of the google-genai Python SDK. The SDK automatically converts the Python functions to the required schema, executes the function calls when requested by the model, and sends the results back to the model to complete the task.


import os
from google import genai
from google.genai import types

# Example Functions
def get_weather_forecast(location: str) -> dict:
    """Gets the current weather temperature for a given location."""
    print(f"Tool Call: get_weather_forecast(location={location})")
    # TODO: Make API call
    print("Tool Response: {'temperature': 25, 'unit': 'celsius'}")
    return {"temperature": 25, "unit": "celsius"}  # Dummy response

def set_thermostat_temperature(temperature: int) -> dict:
    """Sets the thermostat to a desired temperature."""
    print(f"Tool Call: set_thermostat_temperature(temperature={temperature})")
    # TODO: Interact with a thermostat API
    print("Tool Response: {'status': 'success'}")
    return {"status": "success"}

# Configure the client and model
client = genai.Client()
config = types.GenerateContentConfig(
    tools=[get_weather_forecast, set_thermostat_temperature]
)

# Make the request
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="If it's warmer than 20°C in London, set the thermostat to 20°C, otherwise set it to 18°C.",
    config=config,
)

# Print the final, user-facing response
print(response.text)
Expected Output

When you run the code, you will see the SDK orchestrating the function calls. The model first calls get_weather_forecast, receives the temperature, and then calls set_thermostat_temperature with the correct value based on the logic in the prompt.


Tool Call: get_weather_forecast(location=London)
Tool Response: {'temperature': 25, 'unit': 'celsius'}
Tool Call: set_thermostat_temperature(temperature=20)
Tool Response: {'status': 'success'}
OK. I've set the thermostat to 20°C.
Compositional function calling is a native Live API feature. This means Live API can handle the function calling similar to the Python SDK.

Python
JavaScript

# Light control schemas
turn_on_the_lights_schema = {'name': 'turn_on_the_lights'}
turn_off_the_lights_schema = {'name': 'turn_off_the_lights'}

prompt = """
  Hey, can you write run some python code to turn on the lights, wait 10s and then turn off the lights?
  """

tools = [
    {'code_execution': {}},
    {'function_declarations': [turn_on_the_lights_schema, turn_off_the_lights_schema]}
]

await run(prompt, tools=tools, modality="AUDIO")
Function calling modes

The Gemini API lets you control how the model uses the provided tools (function declarations). Specifically, you can set the mode within the.function_calling_config.

AUTO (Default): The model decides whether to generate a natural language response or suggest a function call based on the prompt and context. This is the most flexible mode and recommended for most scenarios.
ANY: The model is constrained to always predict a function call and guarantees function schema adherence. If allowed_function_names is not specified, the model can choose from any of the provided function declarations. If allowed_function_names is provided as a list, the model can only choose from the functions in that list. Use this mode when you require a function call response to every prompt (if applicable).
NONE: The model is prohibited from making function calls. This is equivalent to sending a request without any function declarations. Use this to temporarily disable function calling without removing your tool definitions.
Python
JavaScript

from google.genai import types

# Configure function calling mode
tool_config = types.ToolConfig(
    function_calling_config=types.FunctionCallingConfig(
        mode="ANY", allowed_function_names=["get_current_temperature"]
    )
)

# Create the generation config
config = types.GenerateContentConfig(
    tools=[tools],  # not defined here.
    tool_config=tool_config,
)
Automatic function calling (Python only)

When using the Python SDK, you can provide Python functions directly as tools. The SDK converts these functions into declarations, manages the function call execution, and handles the response cycle for you. Define your function with type hints and a docstring. For optimal results, it is recommended to use Google-style docstrings. The SDK will then automatically:

Detect function call responses from the model.
Call the corresponding Python function in your code.
Send the function's response back to the model.
Return the model's final text response.
The SDK currently does not parse argument descriptions into the property description slots of the generated function declaration. Instead, it sends the entire docstring as the top-level function description.

Python

from google import genai
from google.genai import types

# Define the function with type hints and docstring
def get_current_temperature(location: str) -> dict:
    """Gets the current temperature for a given location.

    Args:
        location: The city and state, e.g. San Francisco, CA

    Returns:
        A dictionary containing the temperature and unit.
    """
    # ... (implementation) ...
    return {"temperature": 25, "unit": "Celsius"}

# Configure the client
client = genai.Client()
config = types.GenerateContentConfig(
    tools=[get_current_temperature]
)  # Pass the function itself

# Make the request
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="What's the temperature in Boston?",
    config=config,
)

print(response.text)  # The SDK handles the function call and returns the final text
You can disable automatic function calling with:

Python

config = types.GenerateContentConfig(
    tools=[get_current_temperature],
    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)
)
Automatic function schema declaration

The API is able to describe any of the following types. Pydantic types are allowed, as long as the fields defined on them are also composed of allowed types. Dict types (like dict[str: int]) are not well supported here, don't use them.

Python

AllowedType = (
  int | float | bool | str | list['AllowedType'] | pydantic.BaseModel)
To see what the inferred schema looks like, you can convert it using from_callable:

Python

def multiply(a: float, b: float):
    """Returns a * b."""
    return a * b

fn_decl = types.FunctionDeclaration.from_callable(callable=multiply, client=client)

# to_json_dict() provides a clean JSON representation.
print(fn_decl.to_json_dict())
Multi-tool use: Combine native tools with function calling

You can enable multiple tools combining native tools with function calling at the same time. Here's an example that enables two tools, Grounding with Google Search and code execution, in a request using the Live API.

Note: Multi-tool use is a-Live API only feature at the moment. The run() function declaration, which handles the asynchronous websocket setup, is omitted for brevity.
Python
JavaScript

# Multiple tasks example - combining lights, code execution, and search
prompt = """
  Hey, I need you to do three things for me.

    1.  Turn on the lights.
    2.  Then compute the largest prime palindrome under 100000.
    3.  Then use Google Search to look up information about the largest earthquake in California the week of Dec 5 2024.

  Thanks!
  """

tools = [
    {'google_search': {}},
    {'code_execution': {}},
    {'function_declarations': [turn_on_the_lights_schema, turn_off_the_lights_schema]} # not defined here.
]

# Execute the prompt with specified tools in audio modality
await run(prompt, tools=tools, modality="AUDIO")
Python developers can try this out in the Live API Tool Use notebook.

Model context protocol (MCP)

Model Context Protocol (MCP) is an open standard for connecting AI applications with external tools and data. MCP provides a common protocol for models to access context, such as functions (tools), data sources (resources), or predefined prompts.

The Gemini SDKs have built-in support for the MCP, reducing boilerplate code and offering automatic tool calling for MCP tools. When the model generates an MCP tool call, the Python and JavaScript client SDK can automatically execute the MCP tool and send the response back to the model in a subsequent request, continuing this loop until no more tool calls are made by the model.

Here, you can find an example of how to use a local MCP server with Gemini and mcp SDK.

Python
JavaScript
Make sure the latest version of the mcp SDK is installed on your platform of choice.


pip install mcp
Note: Python supports automatic tool calling by passing in the ClientSession into the tools parameters. If you want to disable it, you can provide automatic_function_calling with disabled True.

import os
import asyncio
from datetime import datetime
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from google import genai

client = genai.Client()

# Create server parameters for stdio connection
server_params = StdioServerParameters(
    command="npx",  # Executable
    args=["-y", "@philschmid/weather-mcp"],  # MCP Server
    env=None,  # Optional environment variables
)

async def run():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Prompt to get the weather for the current day in London.
            prompt = f"What is the weather in London in {datetime.now().strftime('%Y-%m-%d')}?"

            # Initialize the connection between client and server
            await session.initialize()

            # Send request to the model with MCP function declarations
            response = await client.aio.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    temperature=0,
                    tools=[session],  # uses the session, will automatically call the tool
                    # Uncomment if you **don't** want the SDK to automatically call the tool
                    # automatic_function_calling=genai.types.AutomaticFunctionCallingConfig(
                    #     disable=True
                    # ),
                ),
            )
            print(response.text)

# Start the asyncio event loop and run the main function
asyncio.run(run())
Limitations with built-in MCP support

Built-in MCP support is a experimental feature in our SDKs and has the following limitations:

Only tools are supported, not resources nor prompts
It is available for the Python and JavaScript/TypeScript SDK.
Breaking changes might occur in future releases.
Manual integration of MCP servers is always an option if these limit what you're building.

Send feedback Files API
The Gemini family of artificial intelligence (AI) models is built to handle various types of input data, including text, images, and audio. Since these models can handle more than one type or mode of data, the Gemini models are called multimodal models or explained as having multimodal capabilities.

This guide shows you how to work with media files using the Files API. The basic operations are the same for audio files, images, videos, documents, and other supported file types.

For file prompting guidance, check out the File prompt guide section.

Upload a file

You can use the Files API to upload a media file. Always use the Files API when the total request size (including the files, text prompt, system instructions, etc.) is larger than 20 MB.

The following code uploads a file and then uses the file in a call to generateContent.

Python
JavaScript
Go
REST

from google import genai

client = genai.Client()

myfile = client.files.upload(file="path/to/sample.mp3")

response = client.models.generate_content(
    model="gemini-2.5-flash", contents=["Describe this audio clip", myfile]
)

print(response.text)
Get metadata for a file

You can verify that the API successfully stored the uploaded file and get its metadata by calling files.get.

Python
JavaScript
Go
REST

myfile = client.files.upload(file='path/to/sample.mp3')
file_name = myfile.name
myfile = client.files.get(name=file_name)
print(myfile)
List uploaded files

You can upload multiple files using the Files API. The following code gets a list of all the files uploaded:

Python
JavaScript
Go
REST

print('My files:')
for f in client.files.list():
    print(' ', f.name)
Delete uploaded files

Files are automatically deleted after 48 hours. You can also manually delete an uploaded file:

Python
JavaScript
Go
REST

myfile = client.files.upload(file='path/to/sample.mp3')
client.files.delete(name=myfile.name)
Usage info

You can use the Files API to upload and interact with media files. The Files API lets you store up to 20 GB of files per project, with a per-file maximum size of 2 GB. Files are stored for 48 hours. During that time, you can use the API to get metadata about the files, but you can't download the files. The Files API is available at no cost in all regions where the Gemini API is available.

File prompting strategies

This section provides guidance and best practices for using media files with prompts for the Gemini API.

Being able to use various types of data in your prompts gives you more flexibility in terms of what tasks you can tackle with the Gemini API. For example, you can send the model a photo of a delicious meal and ask it to write a short blog about the meal.

Prompt	Response
Write a short, engaging blog post based on this picture. It should include a description of the meal in the photo and talk about my journey meal prepping.


Cookies	
Meal prepping is a great way to save time and money, and it can also help you to eat healthier. This meal is a great example of a healthy and delicious meal that can be easily prepped ahead of time.

This meal features brown rice, roasted vegetables, and chicken teriyaki. The brown rice is a healthy and complex carbohydrate that will give you sustained energy. The roasted vegetables are a great way to get your daily dose of vitamins and minerals, and the chicken teriyaki is a delicious and protein-rich option.

This meal is also very easy to prepare. Simply cook the brown rice, roast the vegetables, and cook the chicken teriyaki. Once everything is cooked, divide it into meal prep containers and store them in the refrigerator. You can then grab a container and go on busy mornings!

If you're looking for a healthy and delicious meal that can be easily prepped ahead of time, this meal is a great option. It's packed with nutrients and flavor, and it's sure to keep you feeling full and satisfied.

Here's to healthy and delicious meal prepping!
If you are having trouble getting the output you want from prompts that use media files, there are some strategies that can help you get the results you want. The following sections provide design approaches and troubleshooting tips for improving prompts that use multimodal input.

You can improve your multimodal prompts by following these best practices:

Prompt design fundamentals

Be specific in your instructions: Craft clear and concise instructions that leave minimal room for misinterpretation.
Add a few examples to your prompt: Use realistic few-shot examples to illustrate what you want to achieve.
Break it down step-by-step: Divide complex tasks into manageable sub-goals, guiding the model through the process.
Specify the output format: In your prompt, ask for the output to be in the format you want, like markdown, JSON, HTML and more.
Put your image first for single-image prompts: While Gemini can handle image and text inputs in any order, for prompts containing a single image, it might perform better if that image (or video) is placed before the text prompt. However, for prompts that require images to be highly interleaved with texts to make sense, use whatever order is most natural.
Troubles
Structured output
You can configure Gemini for structured output instead of unstructured text, allowing precise extraction and standardization of information for further processing. For example, you can use structured output to extract information from resumes, standardize them to build a structured database.

Gemini can generate either JSON or enum values as structured output.

Generating JSON

There are two ways to generate JSON using the Gemini API:

Configure a schema on the model
Provide a schema in a text prompt
Configuring a schema on the model is the recommended way to generate JSON, because it constrains the model to output JSON.

Configuring a schema (recommended)

To constrain the model to generate JSON, configure a responseSchema. The model will then respond to any prompt with JSON-formatted output.

Python
JavaScript
Go
REST

from google import genai
from pydantic import BaseModel

class Recipe(BaseModel):
    recipe_name: str
    ingredients: list[str]

client = genai.Client()
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="List a few popular cookie recipes, and include the amounts of ingredients.",
    config={
        "response_mime_type": "application/json",
        "response_schema": list[Recipe],
    },
)
# Use the response as a JSON string.
print(response.text)

# Use instantiated objects.
my_recipes: list[Recipe] = response.parsed
Note: Pydantic validators are not yet supported. If a pydantic.ValidationError occurs, it is suppressed, and .parsed may be empty/null.
The output might look like this:


[
  {
    "recipeName": "Chocolate Chip Cookies",
    "ingredients": [
      "1 cup (2 sticks) unsalted butter, softened",
      "3/4 cup granulated sugar",
      "3/4 cup packed brown sugar",
      "1 teaspoon vanilla extract",
      "2 large eggs",
      "2 1/4 cups all-purpose flour",
      "1 teaspoon baking soda",
      "1 teaspoon salt",
      "2 cups chocolate chips"
    ]
  },
  ...
]
Providing a schema in a text prompt

Instead of configuring a schema, you can supply a schema as natural language or pseudo-code in a text prompt. This method is not recommended, because it might produce lower quality output, and because the model is not constrained to follow the schema.

Warning: Don't provide a schema in a text prompt if you're configuring a responseSchema. This can produce unexpected or low quality results.
Here's a generic example of a schema provided in a text prompt:


List a few popular cookie recipes, and include the amounts of ingredients.

Produce JSON matching this specification:

Recipe = { "recipeName": string, "ingredients": array<string> }
Return: array<Recipe>
Since the model gets the schema from text in the prompt, you might have some flexibility in how you represent the schema. But when you supply a schema inline like this, the model is not actually constrained to return JSON. For a more deterministic, higher quality response, configure a schema on the model, and don't duplicate the schema in the text prompt.

Generating enum values

In some cases you might want the model to choose a single option from a list of options. To implement this behavior, you can pass an enum in your schema. You can use an enum option anywhere you could use a string in the responseSchema, because an enum is an array of strings. Like a JSON schema, an enum lets you constrain model output to meet the requirements of your application.

For example, assume that you're developing an application to classify musical instruments into one of five categories: "Percussion", "String", "Woodwind", "Brass", or ""Keyboard"". You could create an enum to help with this task.

In the following example, you pass an enum as the responseSchema, constraining the model to choose the most appropriate option.

Python
JavaScript
REST

from google import genai
import enum

class Instrument(enum.Enum):
  PERCUSSION = "Percussion"
  STRING = "String"
  WOODWIND = "Woodwind"
  BRASS = "Brass"
  KEYBOARD = "Keyboard"

client = genai.Client()
response = client.models.generate_content(
    model='gemini-2.5-flash',
    contents='What type of instrument is an oboe?',
    config={
        'response_mime_type': 'text/x.enum',
        'response_schema': Instrument,
    },
)

print(response.text)
# Woodwind
The Python library will translate the type declarations for the API. However, the API accepts a subset of the OpenAPI 3.0 schema (Schema).

There are two other ways to specify an enumeration. You can use a Literal: ```

Python

Literal["Percussion", "String", "Woodwind", "Brass", "Keyboard"]
And you can also pass the schema as JSON:

Python

from google import genai

client = genai.Client()
response = client.models.generate_content(
    model='gemini-2.5-flash',
    contents='What type of instrument is an oboe?',
    config={
        'response_mime_type': 'text/x.enum',
        'response_schema': {
            "type": "STRING",
            "enum": ["Percussion", "String", "Woodwind", "Brass", "Keyboard"],
        },
    },
)

print(response.text)
# Woodwind
Beyond basic multiple choice problems, you can use an enum anywhere in a JSON schema. For example, you could ask the model for a list of recipe titles and use a Grade enum to give each title a popularity grade:

Python

from google import genai

import enum
from pydantic import BaseModel

class Grade(enum.Enum):
    A_PLUS = "a+"
    A = "a"
    B = "b"
    C = "c"
    D = "d"
    F = "f"

class Recipe(BaseModel):
  recipe_name: str
  rating: Grade

client = genai.Client()
response = client.models.generate_content(
    model='gemini-2.5-flash',
    contents='List 10 home-baked cookie recipes and give them grades based on tastiness.',
    config={
        'response_mime_type': 'application/json',
        'response_schema': list[Recipe],
    },
)

print(response.text)
The response might look like this:


[
  {
    "recipe_name": "Chocolate Chip Cookies",
    "rating": "a+"
  },
  {
    "recipe_name": "Peanut Butter Cookies",
    "rating": "a"
  },
  {
    "recipe_name": "Oatmeal Raisin Cookies",
    "rating": "b"
  },
  ...
]
About JSON schemas

Configuring the model for JSON output using responseSchema parameter relies on Schema object to define its structure. This object represents a select subset of the OpenAPI 3.0 Schema object, and also adds a propertyOrdering field.

Tip: On Python, when you use a Pydantic model, you don't need to directly work with Schema objects, as it gets automatically converted to the corresponding JSON schema. To learn more, see JSON schemas in Python.
Here's a pseudo-JSON representation of all the Schema fields:


{
  "type": enum (Type),
  "format": string,
  "description": string,
  "nullable": boolean,
  "enum": [
    string
  ],
  "maxItems": integer,
  "minItems": integer,
  "properties": {
    string: {
      object (Schema)
    },
    ...
  },
  "required": [
    string
  ],
  "propertyOrdering": [
    string
  ],
  "items": {
    object (Schema)
  }
}
The Type of the schema must be one of the OpenAPI Data Types, or a union of those types (using anyOf). Only a subset of fields is valid for each Type. The following list maps each Type to a subset of the fields that are valid for that type:

string -> enum, format, nullable
integer -> format, minimum, maximum, enum, nullable
number -> format, minimum, maximum, enum, nullable
boolean -> nullable
array -> minItems, maxItems, items, nullable
object -> properties, required, propertyOrdering, nullable
Here are some example schemas showing valid type-and-field combinations:


{ "type": "string", "enum": ["a", "b", "c"] }

{ "type": "string", "format": "date-time" }

{ "type": "integer", "format": "int64" }

{ "type": "number", "format": "double" }

{ "type": "boolean" }

{ "type": "array", "minItems": 3, "maxItems": 3, "items": { "type": ... } }

{ "type": "object",
  "properties": {
    "a": { "type": ... },
    "b": { "type": ... },
    "c": { "type": ... }
  },
  "nullable": true,
  "required": ["c"],
  "propertyOrdering": ["c", "b", "a"]
}
For complete documentation of the Schema fields as they're used in the Gemini API, see the Schema reference.

Property ordering

Warning: When you're configuring a JSON schema, make sure to set propertyOrdering[], and when you provide examples, make sure that the property ordering in the examples matches the schema.
When you're working with JSON schemas in the Gemini API, the order of properties is important. By default, the API orders properties alphabetically and does not preserve the order in which the properties are defined (although the Google Gen AI SDKs may preserve this order). If you're providing examples to the model with a schema configured, and the property ordering of the examples is not consistent with the property ordering of the schema, the output could be rambling or unexpected.

To ensure a consistent, predictable ordering of properties, you can use the optional propertyOrdering[] field.


"propertyOrdering": ["recipeName", "ingredients"]
propertyOrdering[] – not a standard field in the OpenAPI specification – is an array of strings used to determine the order of properties in the response. By specifying the order of properties and then providing examples with properties in that same order, you can potentially improve the quality of results. propertyOrdering is only supported when you manually create types.Schema.

Schemas in Python

When you're using the Python library, the value of response_schema must be one of the following:

A type, as you would use in a type annotation (see the Python typing module)
An instance of genai.types.Schema
The dict equivalent of genai.types.Schema
The easiest way to define a schema is with a Pydantic type (as shown in the previous example):

Python

config={'response_mime_type': 'application/json',
        'response_schema': list[Recipe]}
When you use a Pydantic type, the Python library builds out a JSON schema for you and sends it to the API. For additional examples, see the Python library docs.

The Python library supports schemas defined with the following types (where AllowedType is any allowed type):

int
float
bool
str
list[AllowedType]
AllowedType|AllowedType|...
For structured types:
dict[str, AllowedType]. This annotation declares all dict values to be the same type, but doesn't specify what keys should be included.
User-defined Pydantic models. This approach lets you specify the key names and define different types for the values associated with each of the keys, including nested structures.
JSON Schema support

JSON Schema is a more recent specification than OpenAPI 3.0, which the Schema object is based on. Support for JSON Schema is available as a preview using the field responseJsonSchema which accepts any JSON Schema with the following limitations:

It only works with Gemini 2.5.
While all JSON Schema properties can be passed, not all are supported. See the documentation for the field for more details.
Recursive references can only be used as the value of a non-required object property.
Recursive references are unrolled to a finite degree, based on the size of the schema.
Schemas that contain $ref cannot contain any properties other than those starting with a $.
Here's an example of generating a JSON Schema with Pydantic and submitting it to the model:


curl "https://generativelanguage.googleapis.com/v1alpha/models/\
gemini-2.5-flash:generateContent" \
    -H "x-goog-api-key: $GEMINI_API_KEY"\
    -H 'Content-Type: application/json' \
    -d @- <<EOF
{
  "contents": [{
    "parts":[{
      "text": "Please give a random example following this schema"
    }]
  }],
  "generationConfig": {
    "response_mime_type": "application/json",
    "response_json_schema": $(python3 - << PYEOF
    from enum import Enum
    from typing import List, Optional, Union, Set
    from pydantic import BaseModel, Field, ConfigDict
    import json

    class UserRole(str, Enum):
        ADMIN = "admin"
        VIEWER = "viewer"

    class Address(BaseModel):
        street: str
        city: str

    class UserProfile(BaseModel):
        username: str = Field(description="User's unique name")
        age: Optional[int] = Field(ge=0, le=120)
        roles: Set[UserRole] = Field(min_items=1)
        contact: Union[Address, str]
        model_config = ConfigDict(title="User Schema")

    # Generate and print the JSON Schema
    print(json.dumps(UserProfile.model_json_schema(), indent=2))
    PYEOF
    )
  }
}
EOF
Passing JSON Schema directly is not yet supported when using the SDK.

Best practices

Keep the following considerations and best practices in mind when you're using a response schema:

The size of your response schema counts towards the input token limit.
By default, fields are optional, meaning the model can populate the fields or skip them. You can set fields as required to force the model to provide a value. If there's insufficient context in the associated input prompt, the model generates responses mainly based on the data it was trained on.
A complex schema can result in an InvalidArgument: 400 error. Complexity might come from long property names, long array length limits, enums with many values, objects with lots of optional properties, or a combination of these factors.

If you get this error with a valid schema, make one or more of the following changes to resolve the error:

Shorten property names or enum names.
Flatten nested arrays.
Reduce the number of properties with constraints, such as numbers with minimum and maximum limits.
Reduce the number of properties with complex constraints, such as properties with complex formats like date-time.
Reduce the number of optional properties.
Reduce the number of valid values for enums.
If you aren't seeing the results you expect, add more context to your input prompts or revise your response schema. For example, review the model's response without structured output to see how the model responds. You can then update your response schema so that it better fits the model's output. For additional troubleshooting tips on structured output, see the troubleshooting guide.
What's next

