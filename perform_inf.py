import os
os.chdir("/home/aiops/zhuty/tinyllama")

from pathlib import Path
import transformers
from transformers import AutoModel, AutoTokenizer, LlamaForCausalLM
# model = LlamaForCausalLM.from_pretrained("/home/aiops/zhuty/tinyllama/out/tinyllama_120M/")
# model = LlamaForCausalLM.from_pretrained("/home/aiops/zhuty/tinyllama/out/tinyllaMA_1b/hf_ckpt/")
model = LlamaForCausalLM.from_pretrained("/home/aiops/zhuty/tinyllama/out/tinyllama_120M/tinyllama_120M_20b/")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
## Generate text using this mdoel
from transformers import pipeline
generator = pipeline('text-generation', model=model, tokenizer=tokenizer, return_full_text=False, device="cuda")
# text = "(Shandaken, NY) -- A New York City music teacher who had been missing for over a week is dead after he was found in the woods in Ulster County.\nForty-six-year-old Keith Johnson was last seen May 4th at his Queens school. His vehicle was found at the Woodland Valley Trailhead. Johnson, an avid hiker, was found suffering from hypothermia in the woods of Shandaken on Saturday. Officials say he died before medics were able to get to him. Foul play is not suspected."
# text = "Ever thought about signing up for a a membership with Close to My Heart!? I will give you $25 in select Product credit when you have $300 in sales in the first month! Easy to do with the new Artbooking Cartridge!\n3.Get the Consultant kit and have parties, clubs and crops to sell CTMH as a career!\nBe watching my blog and Facebook for free give aways during the month!"
# text = "Great news for all the variable rate mortgage holders. The Bank of Canada announced this morning that they will not raise the overnight rate which means Prime will remain at 3.45%. You can start your happy dance now.\nThe next scheduled date for announcing the overnight rate target is July 11, 2018."
text = ' steel flange for KKK K-27 turbocharger exhaust side, 3 inch center hole.\nKKK K-27 turbocharger to V-band clamp adapter.\nFor stock KKK K-27 turbo to use V-band clamp. Made with 304 stainless steel.\nFor Tial GT-28R GT-30R GT-35R turbocharger, on Tial turbine housing it is with female side.\nconnector. 50 mm on the side where flanges meet, 54.5 mm on the back of the flange.\n76 mm OD made with 304 stainless steel.\n42mm OD inlet, 51 mm OD outlet, 2 mm thick.\n42mm OD inlet, 52.7 OD outlet, 2 mm thick.'

# returned_seq = generator(text, max_new_tokens=128, num_return_sequences=10, min_new_tokens=3,
#                          num_beams=50, do_sample=True, temperature=1.8, top_k=10000)
# returned_seq = generator(text, max_new_tokens=128, num_return_sequences=10, min_new_tokens=3,
#                          num_beams=50, do_sample=False, temperature=1.8, top_k=10000, num_beam_groups=10, diversity_penalty=2.0,
#                          repetition_penalty=0.5)
pipe = pipeline('text-generation', model=model, device="cuda", tokenizer=tokenizer, return_full_text=False,
                max_new_tokens=64,
                num_return_sequences=10, min_new_tokens=5,
                num_beams=20, do_sample=True, temperature=1.8)

returned_seq = pipe(text)
print("Input: " + text)
# print the returned sequence line by line
for i, seq in enumerate(returned_seq):
    print(f"Generated sequence {i}:")
    print(seq)
    print()