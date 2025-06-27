# template
🚀 This template includes a ready-to-use script for training a high-performing MNIST classifier — but that’s just the beginning!
It's designed with **reproducibility** in mind, without sacrificing **flexibility**

## 🌟 Philosophy
Before diving in, it’s important to understand the philosophy behind this project. In deep learning, it’s easy to get 
swept up in the excitement — experimenting with countless configurations in search of the perfect setup. 🔬✨ 
Eventually, we stumble upon something that works well... only to keep tweaking and lose track of what actually worked 
best. This package is built to help you stay focused, organized, and efficient — so you never lose track of that 
perfect combination again. 🧠✅

The idea is simple: always make your code reproducible! Sure, easier said than done... 😅 My recommended approach is to 
use a multi-level configuration system. Let me explain how it works! 👇

Before jumping into experiments, we usually know the minimal set of parameters required for a project to run. For 
instance, if you're training a Transformer model, you already know you'll need to specify things like the number of 
layers, number of attention heads, learning rate, and so on. All these known parameters can (and should) be stored in a 
configuration file — I personally prefer using YAML for its readability. 📄 When running the experiment, we simply 
load this config file and use it to parameterize each part of the code. Usually, the parameters stored in the config 
gives us the baseline.

Once we’ve established a baseline, it’s natural to want to improve it — whether it's by testing out a new technique 
from a paper or an idea that came to us in a dream. 🚀 But here's the challenge: how do we add new functionality to our 
code without breaking compatibility with earlier runs? In other words, if we use the same config file and script 
parameters, we should still get the exact same results as before. My solution? Add new parameters to functions with 
sensible default values — specifically, defaults that reflect the original behavior. You can then include these 
parameters in your configuration file and toggle them on or off to test their effect. For example, say you’re building 
an image classifier and want to try MixUp. Your training function might look like this:
```python
def train_model(..., use_mixup: bool = False):
    ...
```

By setting the default to False, your baseline run remains intact. Only when use_mixup is explicitly set to True will 
the new logic kick in. This approach ensures clean, reproducible experimentation with minimal disruption. ✅

Sometimes, we don’t want to modify the configuration file directly — for example, when we've decided that a particular 
config represents a fixed setup for a specific model or training strategy. In these cases, it's often more convenient 
to override a few parameters via the command line. 🧪 To do this, I use Python’s built-in argparse module. It adds an 
extra layer of configuration that’s ideal for quick experiments — without changing the original YAML file. And just 
like before, the same principle applies: always use default values that reproduce the results of previous runs. This 
ensures your experiments stay flexible and reproducible. 🔁

This project promotes a simple but powerful principle: make your deep learning experiments reproducible — without 
slowing down iteration or innovation. To achieve that, it recommends a multi-level configuration system:
1. YAML Configuration Files – Store all known parameters for a clean, reproducible baseline. 📄
2. Function Defaults – Add new features with default values that preserve past behavior. This ensures that re-running 
with the same config and cli parameters always gives the same result. ✅
3. CLI Overrides – For quick tweaks, use cli parameters to add new functionalities or to override config's parameters 
without editing the base config. Perfect for fast experimentation. 🧪

This layered setup keeps your workflow organized, traceable, and easy to extend, so you can explore new ideas without
losing sight of what actually works. 🔁

## 🚀 How to use
🧪 Each experiment should be wrapped inside a function named after the experiment itself.\
📄 We recommend placing this function in a file with the same name — keeping things clean and organized!
📝 Alongside it, create a Markdown file (with the same name) to serve as your lab book. Use it to write:
- Your hypothesis
- The goal of the experiment
- Instructions on how to run it
- Or any other things you want to write in your lab book ✅

📊 While you can record your results in the Markdown file, we prefer using a Jupyter notebook for that part — it’s just 
more convenient!

Next, you can register the function inside the `main.py` file to make it available to run 🔥
```python
from experiments.myNewExperiment import myNewExperiment
experiments = {
    ...
    "myNewExperiment": myNewExperiment,
    ...
}
```

To make the code more readable, maintainable and reusable, the template has split the experiment's code into multiple 
files and components.

🚀 Feel free to add CLI arguments in `main.py` to supercharge your code

## Examples
This section shows examples on how to run the template!

## Basic run
▶️ This will run the only experiment included in the template: `experiment1`, using the default configuration out of 
the box.
```shell
python main.py --experiment=experiment --config=configs/config.yml
```

Now, check the training curves with deepboard 📊
```shell
deepboard results/resultTable.db
```

⚠️ Try running the same command again...\
Oops, it won’t work 😕 Why?

That’s because the `resultTable` — the object that tracks all your runs — **blocks duplicate runs by design**.
It helps you avoid rerunning the exact same experiment and wasting resources 💡

However, try running it again by setting the `debug` flag
```shell
python main.py --experiment=experiment --config=configs/config.yml --debug
```

✅ See? Now it runs!

That’s because debug runs are treated differently. Their runID is always -1, and they’re not permanent —
each new debug run simply overwrites the previous one in the resultTable.
🧪 Try it out and check the results in DeepBoard’s UI — it’s a great way to test code quickly without cluttering 
your results!

## Cli parameters
🛠️ We’ve already seen one CLI parameter: `debug`. But that’s just the beginning — there’s so much more you can do!
With CLI arguments, you can easily customize your experiments on the fly: change configs, tweak hyperparameters, switch 
datasets, and more — all without touching the code 🧙‍♂️

Let's say I do not want to select the best model in valid based on the accuracy, but on the loss. I can change
the `watch` parameter
```shell
python main.py --experiment=experiment --config=configs/config.yml --watch=loss
```

Because we changed the cli parameters, another entry was created in the resultTable! This is a powerful way to
quickly try different parameters 🧠 

## Configuration override
🔍 By default, the `resultTable` checks if your configuration file has changed. If it has, a new entry is created in the
 result table. However, when the config file is large, it can be tricky to spot which specific parameters have changed.

That’s why you can override config parameters directly from the CLI! This allows for fine-grained control without 
needing to modify the entire config file.

This code will train for 50 epochs
```shell
python main.py --experiment=experiment --config=configs/config.yml --config.training.num_epochs=50
```

## Your turn
💥 Now it’s your turn! Get in there, **break things**, and really get your hands dirty to make this project 
**yours**! 🤖

🌟 The next breakthrough is yours to make! 😉