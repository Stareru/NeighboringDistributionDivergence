from ndd_compressor import Compressor

compressor = Compressor()


sent="Koishi Komeiji is a character in the series Touhou Project. Satori Komeiji's younger sister. To escape the fear and hatred which other beings feel towards the satori species, she attempted to destroy her mind-reading ability by closing her Third Eye."

biased=False

if biased:
    config={
        "max_itr":5,
        "max_span":9,
        "threshold":1.0,
        "max_len":75,
        "decay_rate":0.9,
        "biased":True,
        "bias_rate":0.9
    }
else:
    config={
        "max_itr":5,
        "max_span":5,
        "threshold":4.0,
        "max_len":75,
        "decay_rate":0.9,
        "biased":False,
        "bias_rate":1.0
    }

logs = compressor.Compress(sent, **config)


for key in logs.keys():
    print(key, logs[key])




