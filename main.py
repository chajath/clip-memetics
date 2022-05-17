import argparse
import os
import pickle

import clip
import kornia
import numpy as np
import torch
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.callback import Callback
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pytorch_pretrained_biggan import (BigGAN, convert_to_images,
                                       display_in_terminal, one_hot_from_names,
                                       save_as_images, truncated_noise_sample)
from torch import nn, optim
from tqdm import tqdm

if torch.cuda.is_available():
    print("CUDA detected")
    device = "cuda"
else:
    print("CUDA not detected, running in CPU mode")
    device = "cpu"

clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)


class ArtGenProblem(Problem):

    def __init__(self, model, goal_embedding, batch_size=10):
        super().__init__(n_var=256, n_obj=1, xl=-2, xu=2)
        self.goal_embedding = goal_embedding
        self.model = model
        self.batch_size = batch_size

    def _evaluate(self, x, out, *args, **argv):
        pop_size = x.shape[0]
        out_F = np.zeros(pop_size)
        for i in range(0, pop_size, self.batch_size):
            end_idx = pop_size if i + self.batch_size > pop_size else i + self.batch_size
            x_batch = x[i:end_idx, :]
            with torch.no_grad():
                imgen_output = self.model.generator(
                    torch.from_numpy(x_batch).type(
                        torch.FloatTensor).to(device), 1)
                imgen_output_resized = kornia.geometry.transform.resize(
                    imgen_output, (224, 224), align_corners=True)
                imgen_embed = clip_model.encode_image(imgen_output_resized)
                loss = -1 * \
                    torch.cosine_similarity(
                        self.goal_embedding, imgen_embed).cpu().numpy()
            z_regloss = 0.1 * np.abs(1 - np.linalg.norm(x_batch[:, :128]))
            c_embedding_regloss = 0.1 * \
                np.abs(1-np.linalg.norm([x_batch[:, 128:]]))
            # TODO: use discriminator score
            out_F[i:end_idx] = loss + z_regloss + c_embedding_regloss
        out["F"] = out_F


class SaveXCallback(Callback):

    def __init__(self) -> None:
        super().__init__()
        self.data["X"] = []

    def notify(self, algorithm):
        self.data["X"].append(algorithm.pop.get("X"))


def run_optimization_save(prompt_text,
                          gan_model,
                          pop_size=10,
                          num_gen=20,
                          random_seed=None,
                          output_base="./data",
                          batch_size=10,
                          sampling=None):
    f_name = f"{prompt_text}-{random_seed}-{pop_size}-{num_gen}"
    print(f_name)
    goal_embedding = clip_model.encode_text(
        clip.tokenize(prompt_text).to(device))
    problem = ArtGenProblem(gan_model, goal_embedding, batch_size=batch_size)
    if sampling is None:
        algorithm = GA(
            pop_size=pop_size,
            seed=random_seed,
        )
    else:
        algorithm = GA(
            pop_size=pop_size,
            seed=random_seed,
            sampling=sampling,
        )
    res = minimize(
        problem,
        algorithm, ('n_gen', num_gen),
        verbose=True,
        seed=random_seed,
        callback=SaveXCallback())
    final_image_gen = gan_model.generator(
        torch.unsqueeze(
            torch.from_numpy(res.X).type(torch.FloatTensor).to(device), 0), 1)
    save_as_images(
        final_image_gen.to('cpu').clone().detach(),
        file_name=f"{output_base}/{f_name}")
    pickle.dump(res, open(f"{output_base}/{f_name}.pt", "wb"))
    return res


def save_images(res, output_base, gan_model, batch_size=10):
    for k, Xs in tqdm(enumerate(res.algorithm.callback.data["X"])):
        pop_size = Xs.shape[0]
        for i in range(0, pop_size, batch_size):
            end_idx = pop_size if i + batch_size > pop_size else i + batch_size
            x_batch = Xs[i:end_idx, :]
            with torch.no_grad():
                final_image_gen = gan_model.generator(
                    torch.from_numpy(x_batch).type(
                        torch.FloatTensor).to(device), 1).detach().to('cpu')
            save_as_images(final_image_gen, f"{output_base}/{k}_{i}")


def main(args):
    os.makedirs(args.output_base, exist_ok=True)
    biggan = BigGAN.from_pretrained(args.biggan_model).to(device)
    # Load previous population.
    if args.from_population is not None:
        with open(args.from_population, "rb") as fp:
            p = pickle.load(fp)
        X = p.algorithm.callback.data["X"][args.from_population_gen]
    else:
        X = None
    res = run_optimization_save(
        args.prompt_text,
        pop_size=args.pop_size,
        num_gen=args.num_gen,
        random_seed=args.random_seed,
        gan_model=biggan,
        output_base=args.output_base,
        batch_size=args.batch_size,
        sampling=X,
    )
    save_images(res, args.output_base, biggan, batch_size=args.batch_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "A main entry point for image generation.")
    parser.add_argument("--prompt_text", type=str, required=True,
                        help="CLIP prompt text for image embeddings to be compared with.")
    parser.add_argument("--pop_size", type=int, default=10,
                        help="Population size of the genetic algorithm.")
    parser.add_argument("--num_gen", type=int, default=20,
                        help="Number of generations to run.")
    parser.add_argument("--random_seed", type=int, default=None,
                        help="Random seed for reproducibility.")
    parser.add_argument("--biggan_model", type=str, default="biggan-deep-512",
                        help="BigGAN model. See https://github.com/huggingface/pytorch-pretrained-BigGAN#models for the list of supported models.")
    parser.add_argument("--clip_model", type=str, default="ViT-B/32",
                        help=f"CLIP model. Available models are: {clip.available_models()}")
    parser.add_argument("--output_base", default="./data",
                        help="Path prefix under which all generated files will be written.")
    parser.add_argument("--batch_size", type=int, default=10,
                        help="Batch size for generation and inference. Tweak this number if you run into trouble with GPU memory issues.")
    parser.add_argument("--from_population", type=str, default=None,
                        help="Path to a pickle dump from which population will be sampled. Will be randomly generated if not given.")
    parser.add_argument("--from_population_gen", type=int, default=-1,
                        help="Generation number of the sampling population. -1 will give the last generation.")
    args = parser.parse_args()
    clip_model, clip_preprocess = clip.load(args.clip_model, device=device)
    main(parser.parse_args())
