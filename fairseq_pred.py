import torch

en2de = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.en-de.single_model')

input_strings = ["Hello Nature", "Today is Wednesday", "+oday is Wednesday",
                  "This is writing team", "This is writing tcam", "Check back later", "Chcck back latcr"]

print("Original   Translated")
for el in input_strings:
  print(el, en2de.translate(el, beam=5))
