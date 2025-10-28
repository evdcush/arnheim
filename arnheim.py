#!/usr/bin/env python
# coding: utf-8

# # Imports

# In[1]:


import torch
import torch.nn as nn
import clip
print("Torch version:", torch.__version__)


# In[2]:


import ray


# In[3]:


import collections
import copy
import math
import time

import cloudpickle

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageDraw
from skimage import transform


# In[4]:


#get_ipython().run_line_magic('pinfo', 'clip.load')


# In[6]:


#clip.available_models()


# In[7]:


from pathlib import Path


# In[10]:


# Load CLIP

CLIP_MODEL = "ViT-B/32"
model_dst_path = Path.home() / 'Models/clip'
model_dst = str(model_dst_path)
#device = torch.device("cuda")
device = torch.device("cuda:1")
print(f"Downloading CLIP model {CLIP_MODEL} to {model_dst}")
#model, _ = clip.load(CLIP_MODEL, device, jit=False, download_root=model_dst)
#model, preproc = clip.load(CLIP_MODEL, device, jit=False, download_root=model_dst)
model, _ = clip.load(CLIP_MODEL, device, jit=False, download_root=model_dst)


# In[11]:


#preproc


# # Neural Visual Grammar

# ### Drawing primitives

# ### ah 2-space indent garb
# goog still do this?
#
# wait, when was arnheim publish...?
# i thought maybe before they get clapped by goog style and
# just have to export to py and fix it:
# jupyter-nbconvert arnheim.ipynb --to python
# pip install reindent
# reindent arnheim.py
#
# fixed

# In[ ]:


def to_homogeneous(p):
    r, c = p
    return np.stack((r, c, np.ones_like(p[0])), axis=0)

def from_homogeneous(p):
    p = p / p.T[:, 2]
    return p[0].astype("int32"), p[1].astype("int32")

def apply_scale(scale, lineh):
    return np.stack([lineh[0, :] * scale,
                     lineh[1, :] * scale,
                     lineh[2, :]])

def apply_translation(translation, lineh, offset_r=0, offset_c=0):
    r, c = translation
    return np.stack([lineh[0, :] + c + offset_c,
                     lineh[1, :] + r + offset_r,
                     lineh[2, :]])

def apply_rotation(translation, rad, lineh):
    r, c = translation
    cos_rad = np.cos(rad)
    sin_rad = np.sin(rad)
    return np.stack(
        [(lineh[0, :] - c) * cos_rad - (lineh[1, :] - r) * sin_rad + c,
         (lineh[0, :] - c) * sin_rad + (lineh[1, :] - r) * cos_rad + r,
         lineh[2, :]])


# In[ ]:


def transform_lines(line_from, line_to, translation, angle, scale,
                    translation2, angle2, scale2, img_siz2):
    """Transform lines by translation, angle and scale, twice.

    Args:
      line_from: Line start point.
      line_to: Line end point.
      translation: 1st translation to line.
      angle: 1st angle of rotation for line.
      scale: 1st scale for line.
      translation2: 2nd translation to line.
      angle2: 2nd angle of rotation for line.
      scale2: 2nd scale for line.
      img_siz2: Offset for 2nd translation.

    Returns:
      Transformed lines.
    """
    if len(line_from.shape) == 1:
        line_from = np.expand_dims(line_from, 0)
    if len(line_to.shape) == 1:
        line_to = np.expand_dims(line_to, 0)

    # First transform.
    line_from_h = to_homogeneous(line_from.T)
    line_to_h = to_homogeneous(line_to.T)
    line_from_h = apply_scale(scale, line_from_h)
    line_to_h = apply_scale(scale, line_to_h)
    translated_line_from = apply_translation(translation, line_from_h)
    translated_line_to = apply_translation(translation, line_to_h)
    translated_mid_point = (translated_line_from + translated_line_to) / 2.0
    translated_mid_point = translated_mid_point[[1, 0]]
    line_from_transformed = apply_rotation(translated_mid_point,
                                           np.pi * angle,
                                           translated_line_from)
    line_to_transformed = apply_rotation(translated_mid_point,
                                         np.pi * angle,
                                         translated_line_to)
    line_from_transformed = np.array(from_homogeneous(line_from_transformed))
    line_to_transformed = np.array(from_homogeneous(line_to_transformed))

    # Second transform.
    line_from_h = to_homogeneous(line_from_transformed)
    line_to_h = to_homogeneous(line_to_transformed)
    line_from_h = apply_scale(scale2, line_from_h)
    line_to_h = apply_scale(scale2, line_to_h)
    translated_line_from = apply_translation(
        translation2, line_from_h, offset_r=img_siz2, offset_c=img_siz2)
    translated_line_to = apply_translation(
        translation2, line_to_h, offset_r=img_siz2, offset_c=img_siz2)
    translated_mid_point = (translated_line_from + translated_line_to) / 2.0
    translated_mid_point = translated_mid_point[[1, 0]]
    line_from_transformed = apply_rotation(translated_mid_point,
                                           np.pi * angle2,
                                           translated_line_from)
    line_to_transformed = apply_rotation(translated_mid_point,
                                         np.pi * angle2,
                                         translated_line_to)
    return np.concatenate([from_homogeneous(line_from_transformed),
                           from_homogeneous(line_to_transformed)],
                          axis=1)


# ### Hierarchical stroke painting functions

# In[ ]:


# PaintingCommand
#   origin_top: Origin of line defined by top level LSTM
#   angle_top: Angle of line defined by top level LSTM
#   scale_top: Scale for line defined by top level LSTM
#   origin_bottom: Origin of line defined by bottom level LSTM
#   angle_bottom: Angle of line defined by bottom level LSTM
#   scale_bottom: Scale for line defined by bottom level LSTM
#   position_choice: Selects between use of:
#     Origin, angle and scale from both LSTM levels
#     Origin, angle and scale just from top level LSTM
#     Origin, angle and scale just from bottom level LSTM
#   transparency: Line transparency determined by bottom level LSTM
PaintingCommand = collections.namedtuple("PaintingCommand",
                                         ["origin_top",
                                          "angle_top",
                                          "scale_top",
                                          "origin_bottom",
                                          "angle_bottom",
                                          "scale_bottom",
                                          "position_choice",
                                          "transparency"])

def paint_over_image(img, strokes, painting_commands,
                     allow_strokes_beyond_image_edges, coeff_size=1):
    """Make marks over an existing image.

    Args:
      img: Image to draw on.
      strokes: Stroke descriptions.
      painting_commands: Top-level painting commands with transforms for the i
        sets of strokes.
      allow_strokes_beyond_image_edges: Allow strokes beyond image boundary.
      coeff_size: Determines low res (1) or high res (10) image will be drawn.

    Returns:
      num_strokes: The number of strokes made.
    """
    img_center = 112. * coeff_size
    # a, b and c: determines the stroke width distribution (see 'weights' below)
    a = 10. * coeff_size
    b = 2. * coeff_size
    c = 300. * coeff_size
    # d: extent that the strokes are allowed to go beyond the edge of the canvas
    d = 223 * coeff_size

    def _clip_colour(col):
        return np.clip((np.round(col * 255. + 128.)).astype(np.int32), 0, 255)

    # Loop over all the top level...
    t0_over = time.time()
    num_strokes = sum(len(s) for s in strokes)
    translations = np.zeros((2, num_strokes,), np.float32)
    translations2 = np.zeros((2, num_strokes,), np.float32)
    angles = np.zeros((num_strokes,), np.float32)
    angles2 = np.zeros((num_strokes,), np.float32)
    scales = np.zeros((num_strokes,), np.float32)
    scales2 = np.zeros((num_strokes,), np.float32)
    weights = np.zeros((num_strokes,), np.float32)
    lines_from = np.zeros((num_strokes, 2), np.float32)
    lines_to = np.zeros((num_strokes, 2), np.float32)
    rgbas = np.zeros((num_strokes, 4), np.float32)
    k = 0
    for i in range(len(strokes)):

        # Get the top-level transforms for the i-th bunch of strokes
        painting_comand = painting_commands[i]
        translation_a = painting_comand.origin_top
        angle_a = (painting_comand.angle_top + 1) / 5.0
        scale_a = 0.5 + (painting_comand.scale_top + 1) / 3.0
        translation_b = painting_comand.origin_bottom
        angle_b = (painting_comand.angle_bottom + 1) / 5.0
        scale_b = 0.5 + (painting_comand.scale_bottom + 1) / 3.0
        position_choice = painting_comand.position_choice
        solid_colour = painting_comand.transparency

        # Do we use origin, angle and scale from both, top or bottom LSTM levels?
        if position_choice > 0.33:
            translation = translation_a
            angle = angle_a
            scale = scale_a
            translation2 = translation_b
            angle2 = angle_b
            scale2 = scale_b
        elif position_choice > -0.33:
            translation = translation_a
            angle = angle_a
            scale = scale_a
            translation2 = [-img_center, -img_center]
            angle2 = 0.
            scale2 = 1.
        else:
            translation = translation_b
            angle = angle_b
            scale = scale_b
            translation2 = [-img_center, -img_center]
            angle2 = 0.
            scale2 = 1.

        # Store top-level transforms
        strokes_i = strokes[i]
        n_i = len(strokes_i)
        angles[k:(k+n_i)] = angle
        angles2[k:(k+n_i)] = angle2
        scales[k:(k+n_i)] = scale
        scales2[k:(k+n_i)] = scale2
        translations[0, k:(k+n_i)] = translation[0]
        translations[1, k:(k+n_i)] = translation[1]
        translations2[0, k:(k+n_i)] = translation2[0]
        translations2[1, k:(k+n_i)] = translation2[1]

        # ... and the bottom level stroke definitions.
        for j in range(n_i):
            z_ij = strokes_i[j]

            # Store line weight (we will process micro-strokes later)
            weights[k] = z_ij[4]
            # Store line endpoints
            lines_from[k, :] = (z_ij[0], z_ij[1])
            lines_to[k, :] = (z_ij[2], z_ij[3])

            # Store colour and alpha
            rgbas[k, 0] = z_ij[7]
            rgbas[k, 1] = z_ij[8]
            rgbas[k, 2] = z_ij[9]
            if solid_colour > -0.5:
                rgbas[k, 3] = 25.5
            else:
                rgbas[k, 3] = z_ij[11]
            k += 1

    # Draw all the strokes in a batch as sequence of length 2 * num_strokes
    t1_over = time.time()
    lines_from *= img_center/2.0
    lines_to *= img_center/2.0
    rr, cc = transform_lines(lines_from, lines_to, translations, angles, scales,
                             translations2, angles2, scales2, img_center)
    if not allow_strokes_beyond_image_edges:
        rrm = np.round(np.clip(rr, 1, d-1)).astype(int)
        ccm = np.round(np.clip(cc, 1, d-1)).astype(int)
    else:
        rrm = np.round(rr).astype(int)
        ccm = np.round(cc).astype(int)

    # Plot all the strokes
    t2_over = time.time()
    img_pil = Image.fromarray(img)
    canvas = ImageDraw.Draw(img_pil, "RGBA")
    rgbas[:, :3] = _clip_colour(rgbas[:, :3])
    rgbas[:, 3] = (np.clip(5.0 * np.abs(rgbas[:, 3]), 0, 255)).astype(np.int32)
    weights = (np.clip(np.round(weights * b + a), 2, c)).astype(np.int32)
    for k in range(num_strokes):
        canvas.line((rrm[k], ccm[k], rrm[k+num_strokes], ccm[k+num_strokes]),
                    fill=tuple(rgbas[k]), width=weights[k])
    img[:] = np.asarray(img_pil)[:]
    t3_over = time.time()
    if VERBOSE_CODE:
        print("{:.2f}s to store {} stroke defs, {:.4f}s to "
              "compute them, {:.4f}s to plot them".format(
                  t1_over - t0_over, num_strokes, t2_over - t1_over,
                  t3_over - t2_over))
    return num_strokes


# ### Recurrent Neural Network Layer Generator

# In[ ]:


# DrawingLSTMSpec - parameters defining the LSTM architecture
#   input_spec_size: Size if sequence elements
#   num_lstms: Number of LSTMs at each layer
#   net_lstm_hiddens: Number of hidden LSTM units
#   net_mlp_hiddens: Number of hidden units in MLP layer
DrawingLSTMSpec = collections.namedtuple("DrawingLSTMSpec",
                                         ["input_spec_size",
                                          "num_lstms",
                                          "net_lstm_hiddens",
                                          "net_mlp_hiddens"])


class MakeGeneratorLstm(nn.Module):
    """Block of parallel LSTMs with MLP output heads."""

    def __init__(self, drawing_lstm_spec, output_size):
        """Build drawing LSTM architecture using spec.

        Args:
          drawing_lstm_spec: DrawingLSTMSpec with architecture parameters
          output_size: Number of outputs for the MLP head layer
        """
        super(MakeGeneratorLstm, self).__init__()
        self._num_lstms = drawing_lstm_spec.num_lstms
        self._input_layer = nn.Sequential(
            nn.Linear(drawing_lstm_spec.input_spec_size,
                      drawing_lstm_spec.net_lstm_hiddens),
            torch.nn.LeakyReLU(0.2, inplace=True))
        lstms = []
        heads = []
        for _ in range(self._num_lstms):
            lstm_layer = nn.LSTM(
                input_size=drawing_lstm_spec.net_lstm_hiddens,
                hidden_size=drawing_lstm_spec.net_lstm_hiddens,
                num_layers=2, batch_first=True, bias=True)
            head_layer = nn.Sequential(
                nn.Linear(drawing_lstm_spec.net_lstm_hiddens,
                          drawing_lstm_spec.net_mlp_hiddens),
                torch.nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(drawing_lstm_spec.net_mlp_hiddens, output_size))
            lstms.append(lstm_layer)
            heads.append(head_layer)
        self._lstms = nn.ModuleList(lstms)
        self._heads = nn.ModuleList(heads)

    def forward(self, x):
        pred = []
        x = self._input_layer(x)*10.0
        for i in range(self._num_lstms):
            y, _ = self._lstms[i](x)
            y = self._heads[i](y)
            pred.append(y)
        return pred


# ### DrawingLSTM - A Drawing Recurrent Neural Network

# In[ ]:


Genotype = collections.namedtuple("Genotype",
                                  ["top_lstm",
                                   "bottom_lstm",
                                   "input_sequence",
                                   "initial_img"])

class DrawingLSTM:
    """LSTM for processing input sequences and generating resultant drawings.

    Comprised of two LSTM layers.
    """

    def __init__(self, drawing_lstm_spec, allow_strokes_beyond_image_edges):
        """Create DrawingLSTM to interpret input sequences and paint an image.

        Args:
          drawing_lstm_spec: DrawingLSTMSpec with LSTM architecture parameters
          allow_strokes_beyond_image_edges: Draw lines outside image boundary
        """
        self._input_spec_size = drawing_lstm_spec.input_spec_size
        self._num_lstms = drawing_lstm_spec.num_lstms
        self._allow_strokes_beyond_image_edges = allow_strokes_beyond_image_edges
        with torch.no_grad():
            self.top_lstm = MakeGeneratorLstm(drawing_lstm_spec,
                                              self._input_spec_size)
            self.bottom_lstm = MakeGeneratorLstm(drawing_lstm_spec, 12)
        self._init_all(self.top_lstm, torch.nn.init.normal_, mean=0., std=0.2)
        self._init_all(self.bottom_lstm, torch.nn.init.normal_, mean=0., std=0.2)

    def _init_all(self, a_model, init_func, *params, **kwargs):
        """Method for initialising model with given init_func, params and kwargs."""
        for p in a_model.parameters():
            init_func(p, *params, **kwargs)

    def _feed_top_lstm(self, input_seq):
        """Feed all input sequences input_seq into the LSTM models."""

        x_in = input_seq.reshape((len(input_seq), 1, self._input_spec_size))
        x_in = np.tile(x_in, (SEQ_LENGTH, 1))
        x_torch = torch.from_numpy(x_in).type(torch.FloatTensor)
        y_torch = self.top_lstm(x_torch)
        y_torch = [y_torch_k.detach().numpy() for y_torch_k in y_torch]
        del x_in
        del x_torch

        # There are multiple LSTM heads. For each sequence, read out the head and
        # length of intermediary output to keep and return intermediary outputs.
        readouts_top = np.clip(
            np.round(self._num_lstms/2.0 * (1 + input_seq[:, 1])).astype(np.int32),
            0, self._num_lstms-1)
        lengths_top = np.clip(
            np.round(10.0 * (1 + input_seq[:, 0])).astype(np.int32),
            0, SEQ_LENGTH) + 1
        intermediate_strings = []
        for i in range(len(readouts_top)):
            y_torch_i = y_torch[readouts_top[i]][i]
            intermediate_strings.append(y_torch_i[0:lengths_top[i], :])
        return intermediate_strings

    def _feed_bottom_lstm(self, intermediate_strings, input_seq, coeff_size=1):
        """Feed all input sequences into the LSTM models.

        Args:
          intermediate_strings: top level strings
          input_seq: input sequences fed to the top LSTM
          coeff_size: sets centre origin

        Returns:
          strokes: Painting strokes.
          painting_commands: Top-level painting commands with origin, angle and scale
            information, as well as transparency.
        """
        img_center = 112. * coeff_size
        coeff_origin = 100. * coeff_size
        top_lengths = []
        for i in range(len(intermediate_strings)):
            top_lengths.append(len(intermediate_strings[i]))
        y_flat = np.concatenate(intermediate_strings, axis=0)
        tiled_y_flat = y_flat.reshape((len(y_flat), 1, self._input_spec_size))
        tiled_y_flat = np.tile(tiled_y_flat, (SEQ_LENGTH, 1))
        y_torch = torch.from_numpy(tiled_y_flat).type(torch.FloatTensor)
        z_torch = self.bottom_lstm(y_torch)
        z_torch = [z_torch_k.detach().numpy() for z_torch_k in z_torch]
        del tiled_y_flat
        del y_torch

        # There are multiple LSTM heads. For each sequence, read out the head and
        # length of intermediary output to keep and return intermediary outputs.
        readouts = np.clip(np.round(
            NUM_LSTMS/2.0 * (1 + y_flat[:, 0])).astype(np.int32), 0, NUM_LSTMS-1)
        lengths_bottom = np.clip(
            np.round(10.0 * (1 + y_flat[:, 1])).astype(np.int32), 0, SEQ_LENGTH) + 1
        strokes = []
        painting_commands = []
        offset = 0
        for i in range(len(intermediate_strings)):
            origin_top = [(1+input_seq[i, 2]) * img_center,
                          (1+input_seq[i, 3]) * img_center]
            angle_top = input_seq[i, 4]
            scale_top = input_seq[i, 5]
            for j in range(len(intermediate_strings[i])):
                k = j + offset
                z_torch_ij = z_torch[readouts[k]][k]
                strokes.append(z_torch_ij[0:lengths_bottom[k], :])
                y_ij = y_flat[k]
                origin_bottom = [y_ij[2] * coeff_origin, y_ij[3] * coeff_origin]
                angle_bottom = y_ij[4]
                scale_bottom = y_ij[5]
                position_choice = y_ij[6]
                transparency = y_ij[7]
                painting_command = PaintingCommand(
                    origin_top, angle_top, scale_top, origin_bottom, angle_bottom,
                    scale_bottom, position_choice, transparency)
                painting_commands.append(painting_command)
            offset += top_lengths[i]
        del y_flat
        return strokes, painting_commands

    def make_initial_genotype(self, initial_img, sequence_length,
                              input_spec_size):
        """Make and return initial DNA weights for LSTMs, input sequence, and image.

        Args:
          initial_img: Image (to be appended to the genotype)
          sequence_length: Length of the input sequence (i.e. number of strokes)
          input_spec_size: Number of inputs for each element in the input sequences
        Returns:
          Genotype NamedTuple with fields: [parameters of network 0,
                                            parameters of network 1,
                                            input sequence,
                                            initial_img]
        """
        dna_top = []
        with torch.no_grad():
            for _, params in self.top_lstm.named_parameters():
                dna_top.append(params.clone())
                param_size = params.numpy().shape
                dna_top[-1] = np.random.uniform(
                    0.1 * DNA_SCALE, 0.3
                    * DNA_SCALE) * np.random.normal(size=param_size)
        dna_bottom = []
        with torch.no_grad():
            for _, params in self.bottom_lstm.named_parameters():
                dna_bottom.append(params.clone())
                param_size = params.numpy().shape
                dna_bottom[-1] = np.random.uniform(
                    0.1 * DNA_SCALE, 0.3
                    * DNA_SCALE) * np.random.normal(size=param_size)
        input_sequence = np.random.uniform(
            -1, 1, size=(sequence_length, input_spec_size))
        return Genotype(dna_top, dna_bottom, input_sequence, initial_img)

    def draw(self, img, genotype):
        """Add to the image using the latest genotype and get latest input sequence.

        Args:
          img: image to add to.
          genotype: as created by make_initial_genotype.

        Returns:
          image with new strokes added.
        """
        t0_draw = time.time()
        img = img + genotype.initial_img
        input_sequence = genotype.input_sequence

        # Generate the strokes for drawing in batch mode.
        # input_sequence is between 10 and 20 but is evolved, can go to 200.
        intermediate_strings = self._feed_top_lstm(input_sequence)
        strokes, painting_commands = self._feed_bottom_lstm(
            intermediate_strings, input_sequence)
        del intermediate_strings

        # Now we can go through the output strings producing the strokes.
        t1_draw = time.time()
        num_strokes = paint_over_image(
            img, strokes, painting_commands, self._allow_strokes_beyond_image_edges,
            coeff_size=1)

        t2_draw = time.time()
        if VERBOSE_CODE:
            print(
                "Draw {:.2f}s (net {:.2f}s plot {:.2f}s {:.1f}ms/strk {}".format(
                    t2_draw - t0_draw, t1_draw - t0_draw, t2_draw - t1_draw,
                    (t2_draw - t1_draw) / num_strokes * 1000, num_strokes))
        return img


# ## DrawingGenerator

# In[ ]:


class DrawingGenerator:
    """Creates a drawing using a DrawingLSTM."""

    def __init__(self, image_size, drawing_lstm_spec,
                 allow_strokes_beyond_image_edges):
        self.primitives = ["c", "r", "l", "b", "p", "j"]
        self.pop = []
        self.size = image_size
        self.fitnesses = np.zeros(1)
        self.noise = 2
        self.mutation_std = 0.0004
        # input_spec_size, num_lstms, net_lstm_hiddens,
        # net_mlp_hiddens, output_size, allow_strokes_beyond_image_edges
        self.drawing_lstm = DrawingLSTM(drawing_lstm_spec,
                                        allow_strokes_beyond_image_edges)

    def make_initial_genotype(self, initial_img, sequence_length, input_spec_size):
        """Use drawing_lstm to create initial genotypye."""

        self.genotype = self.drawing_lstm.make_initial_genotype(
            initial_img, sequence_length, input_spec_size)
        return self.genotype


    def _copy_genotype_to_generator(self, genotype):
        """Copy genotype's data into generator's parameters.

        Copies the parameters in genotype (genotype.top_lstm[:] and
        genotype.bottom_lstm[:]) into the parmaters for the drawing network so it
        can be used to evaluate the genotype.

        Args:
          genotype: as created by make_initial_genotype.

        Returns:
          None
        """
        self.genotype = copy.deepcopy(genotype)
        i = 0
        with torch.no_grad():
            for _, param in self.drawing_lstm.top_lstm.named_parameters():
                param.copy_(torch.tensor(self.genotype.top_lstm[i]))
                i = i + 1
        i = 0
        with torch.no_grad():
            for _, param in self.drawing_lstm.bottom_lstm.named_parameters():
                param.copy_(torch.tensor(self.genotype.bottom_lstm[i]))
                i = i + 1

    def _interpret_genotype(self, genotype):
        img = np.zeros((self.size, self.size, 3), dtype=np.uint8)
        img = self.drawing_lstm.draw(img, genotype)
        return img

    def draw_from_genotype(self, genotype):
        """Copy input sequence and LSTM weights from `genotype`, run and draw."""
        self._copy_genotype_to_generator(genotype)
        return self._interpret_genotype(self.genotype)

    def visualize_genotype(self, genotype):
        """Plot histograms of genotype"s data."""

        plt.show()
        inp_seq = np.array(genotype.input_sequence).flatten()
        plt.title("input seq")
        plt.hist(inp_seq)
        plt.show()

        inp_seq = np.array(genotype.top_lstm).flatten()
        plt.title("LSTM top")
        plt.hist(inp_seq)
        plt.show()

        inp_seq = np.array(genotype.bottom_lstm).flatten()
        plt.title("LSTM bottom")
        plt.hist(inp_seq)

        plt.show()

    def mutate(self, genotype):
        """Mutates `genotype`. This function is static.

        Args:
          genotype: genotype structure to mutate parameters of.

        Returns:
          new_genotype: Mutated copy of supplied genotype.
        """

        new_genotype = copy.deepcopy(genotype)
        new_input_seq = new_genotype.input_sequence
        n = len(new_input_seq)

        if np.random.uniform() < 1.0:

            # Standard gaussian small mutation of input sequence.
            if np.random.uniform() > 0.5:
                new_input_seq += (
                    np.random.uniform(0.001, 0.2) * np.random.normal(
                        size=new_input_seq.shape))

            # Low frequency large mutation of individual parts of the input sequence.
            for i in range(n):
                if np.random.uniform() < 2.0/n:
                    for j in range(len(new_input_seq[i])):
                        if np.random.uniform() < 2.0/len(new_input_seq[i]):
                            new_input_seq[i][j] = new_input_seq[i][j] + 0.5*np.random.normal()

            # Adding and deleting elements from the input sequence.
            if np.random.uniform() < 0.01:
                if VERBOSE_MUTATION:
                    print("Mutation: adding")
                a = np.random.uniform(-1, 1, size=(1, INPUT_SPEC_SIZE))
                pos = np.random.randint(1, len(new_input_seq))
                new_input_seq = np.insert(new_input_seq, pos, a, axis=0)
            if np.random.uniform() < 0.02:
                if VERBOSE_MUTATION:
                    print("Mutation: deleting")
                pos = np.random.randint(1, len(new_input_seq))
                new_input_seq = np.delete(new_input_seq, pos, axis=0)
            n = len(new_input_seq)

            # Swapping two elements in the input sequence.
            if np.random.uniform() < 0.01:
                element1 = np.random.randint(0, n)
                element2 = np.random.randint(0, n)
                while element1 == element2:
                    element2 = np.random.randint(0, n)
                temp = copy.deepcopy(new_input_seq[element1])
                new_input_seq[element1] = copy.deepcopy(new_input_seq[element2])
                new_input_seq[element2] = temp

            # Duplicate an element in the input sequence (with some mutation).
            if np.random.uniform() < 0.01:
                if VERBOSE_MUTATION:
                    print("Mutation: duplicating")
                element1 = np.random.randint(0, n)
                element2 = np.random.randint(0, n)
                while element1 == element2:
                    element2 = np.random.randint(0, n)
                new_input_seq[element1] = copy.deepcopy(new_input_seq[element2])
                noise = 0.05 * np.random.normal(size=new_input_seq[element1].shape)
                new_input_seq[element1] += noise

            # Ensure that the input sequence is always between -1 and 1
            # so that positions make sense.
            new_genotype = new_genotype._replace(
                input_sequence=np.clip(new_input_seq, -1.0, 1.0))

        # Mutates dna of networks.
        if np.random.uniform() < 1.0:
            for net in range(2):
                for layer in range(len(new_genotype[net])):
                    weights = new_genotype[net][layer]
                    if np.random.uniform() < 0.5:
                        noise = 0.00001 * np.random.standard_cauchy(size=weights.shape)
                        weights += noise
                    else:
                        noise = np.random.normal(size=weights.shape)
                        noise *= np.random.uniform(0.0001, 0.006)
                        weights += noise

                    if np.random.uniform() < 0.01:
                        noise = np.random.normal(size=weights.shape)
                        noise *= np.random.uniform(0.1, 0.3)
                        weights = noise

                    # Ensure weights are between -10 and 10.
                    weights = np.clip(weights, -1.0, 1.0)
                    new_genotype[net][layer] = weights

        return new_genotype


# ## Evaluator

# In[ ]:


class Evaluator:
    """Evaluator for a drawing."""

    def __init__(self, image_size, drawing_lstm_spec,
                 allow_strokes_beyond_image_edges):
        self.drawing_generator = DrawingGenerator(image_size, drawing_lstm_spec,
                                                  allow_strokes_beyond_image_edges)
        self.calls = 0

    def make_initial_genotype(self, img, sequence_length, input_spec_size):
        return self.drawing_generator.make_initial_genotype(img, sequence_length,
                                                            input_spec_size)

    def evaluate_genotype(self, pickled_genotype, id_num):
        """Evaluate genotype and return genotype's image.

        Args:
          pickled_genotype: pickled genotype to be evaluated.
          id_num: ID number of genotype.

        Returns:
          dict: drawing and id_num.
        """

        genotype = cloudpickle.loads(pickled_genotype)
        drawing = self.drawing_generator.draw_from_genotype(genotype)
        self.calls += 1
        return {"drawing": drawing, "id": id_num}

    def mutate(self, genotype):
        """Create a mutated version of genotype."""
        return self.drawing_generator.mutate(genotype)


# # Evolution

# ## Fitness calculation, tournament, and crossover

# In[ ]:


IMAGE_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).cuda()
IMAGE_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).cuda()

def get_fitness(pictures, use_projective_transform,
                projective_transform_coefficient):
    """Run CLIP on a batch of `pictures` and return `fitnesses`.

    Args:
      pictures: batch if images to evaluate
      use_projective_transform: Add transformed versions of the image
      projective_transform_coefficient: Degree of transform

    Returns:
      Similarities between images and the text
    """

    # Do we use projective transforms of images before CLIP eval?
    t0 = time.time()
    pictures_trans = np.swapaxes(np.array(pictures), 1, 3) / 244.0
    if use_projective_transform:
        for i in range(len(pictures_trans)):
            matrix = np.eye(3) + (
                projective_transform_coefficient * np.random.normal(size=(3, 3)))
            tform = transform.ProjectiveTransform(matrix=matrix)
            pictures_trans[i] = transform.warp(pictures_trans[i], tform.inverse)

    # Run the CLIP evaluator.
    t1 = time.time()
    image_input = torch.tensor(np.stack(pictures_trans)).cuda()
    image_input -= IMAGE_MEAN[:, None, None]
    image_input /= IMAGE_STD[:, None, None]
    with torch.no_grad():
        image_features = model.encode_image(image_input).float()
    t2 = time.time()
    similarity = torch.cosine_similarity(
        text_features, image_features, dim=1).cpu().numpy()
    t3 = time.time()
    if VERBOSE_CODE:
        print(f"get_fitness init {t1-t0:.4f}s, CLIP {t2-t1:.4f}s, sim {t3-t2:.4f}s")
    return similarity


def crossover(dna_winner, dna_loser, crossover_prob):
    """Create new genotype by combining two genotypes.

    Randomly replaces parts of the genotype 'dna_winner' with parts of dna_loser
    to create a new genotype based mostly on on both 'parents'.

    Args:
      dna_winner: The high-fitness parent genotype - gets replaced with child.
      dna_loser: The lower-fitness parent genotype.
      crossover_prob: Probability of crossover between winner and loser.

    Returns:
      dna_winner: The result of crossover from parents.
    """

    # Copy single input signals
    for i in range(len(dna_winner[2])):
        if i < len(dna_loser[2]):
            if np.random.uniform() < crossover_prob:
                dna_winner[2][i] = copy.deepcopy(dna_loser[2][i])

    # Copy whole modules
    for i in range(len(dna_winner[0])):
        if i < len(dna_loser[0]):
            if np.random.uniform() < crossover_prob:
                dna_winner[0][i] = copy.deepcopy(dna_loser[0][i])

    # Copy whole modules
    for i in range(len(dna_winner[1])):
        if i < len(dna_loser[1]):
            if np.random.uniform() < crossover_prob:
                dna_winner[1][i] = copy.deepcopy(dna_loser[1][i])

    return dna_winner


def truncation_selection(population, fitnesses, evaluator, use_crossover,
                         crossover_prob):
    """Create new population using truncation selection.

    Creates a new population by copying across the best 50% genotypes and
    filling the rest with (for use_crossover==False) a mutated copy of each
    genotype or (for use_crossover==True) with children created through crossover
    between each winner and a genotype in the bottom 50%.

    Args:
      population: list of current population genotypes.
      fitnesses: list of evaluated fitnesses.
      evaluator: class that evaluates a draw generator.
      use_crossover: Whether to use crossover between winner and loser.
      crossover_prob: Probability of crossover between winner and loser.

    Returns:
      new_pop: the new population.
      best: genotype.
    """

    fitnesses = np.array(-fitnesses)
    ordered_fitness_ids = fitnesses.argsort()
    best = copy.deepcopy(population[ordered_fitness_ids[0]])
    pop_size = len(population)

    if not use_crossover:
        new_pop = []
        for i in range(int(pop_size/2)):
            new_pop.append(copy.deepcopy(population[ordered_fitness_ids[i]]))
        for i in range(int(pop_size/2)):
            new_pop.append(evaluator.mutate(
                copy.deepcopy(population[ordered_fitness_ids[i]])))
    else:
        new_pop = []
        for i in range(int(pop_size/2)):
            new_pop.append(copy.deepcopy(population[ordered_fitness_ids[i]]))
        for i in range(int(pop_size/2)):
            new_pop.append(evaluator.mutate(crossover(
                copy.deepcopy(population[ordered_fitness_ids[i]]),
                population[ordered_fitness_ids[int(pop_size/2) + i]], crossover_prob
                )))

    return new_pop, best


# ##Remote workers

# In[ ]:


VERBOSE_DURATION = False

@ray.remote
class Worker(object):
    """Takes a pickled dna and evaluates it, returning result."""

    def __init__(self, image_size, drawing_lstm_spec,
                 allow_strokes_beyond_image_edges):
        self.evaluator = Evaluator(image_size, drawing_lstm_spec,
                                   allow_strokes_beyond_image_edges)

    def compute(self, dna_pickle, genotype_id):
        if VERBOSE_DURATION:
            t0 = time.time()
        res = self.evaluator.evaluate_genotype(dna_pickle, genotype_id)
        if VERBOSE_DURATION:
            duration = time.time() - t0
            print(f"Worker {genotype_id} evaluated params in {duration:.1f}sec")
        return res


def create_workers(num_workers, image_size, drawing_lstm_spec,
                   allow_strokes_beyond_image_edges):
    """Create the workers.

    Args:
      num_workers: Number of parallel workers for evaluation.
      image_size: Length of side of (square) image
      drawing_lstm_spec: DrawingLSTMSpec for LSTM network
      allow_strokes_beyond_image_edges: Whether to draw outside the edges
    Returns:
      List of workers.
    """
    worker_pool = []
    for w_i in range(num_workers):
        print("Creating worker", w_i, flush=True)
        worker_pool.append(Worker.remote(image_size, drawing_lstm_spec,
                                         allow_strokes_beyond_image_edges))
    return worker_pool


# ##Plotting

# In[ ]:


def plot_training_res(batch_drawings, fitness_history, idx=None):
    """Plot fitnesses and timings.

    Args:
      batch_drawings: Drawings
      fitness_history: History of fitnesses
      idx: Index of drawing to show, default is highest fitness
    """
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    if idx is None:
        idx = np.argmax(fitness_history[-1])
    ax1.plot(fitness_history, ".")
    ax1.set_title("Fitnesses")
    ax2.imshow(batch_drawings[idx])
    ax2.set_title(f"{PROMPT} (fit: {fitness_history[-1][idx]:.3f})")
    plt.show()

def plot_samples(batch_drawings, num_samples=16):
    """Plot sample of drawings.

    Args:
      batch_drawings: Batch of drawings to sample from
      num_samples: Number to displa
    """
    num_samples = min(len(batch_drawings), num_samples)
    num_rows = int(math.floor(np.sqrt(num_samples)))
    num_cols = int(math.ceil(num_samples / num_rows))
    row_images = []
    for c in range(0, num_samples, num_cols):
        if c + num_cols <= num_samples:
            row_images.append(np.concatenate(batch_drawings[c:(c+num_cols)], axis=1))
    composite_image = np.concatenate(row_images, axis=0)
    _, ax = plt.subplots(1, 1, figsize=(20, 20))
    ax.imshow(composite_image)
    ax.set_title(PROMPT)


# ## Population and evolution main loop

# In[ ]:


def make_population(pop_size, evaluator, image_size, input_spec_size,
                    sequence_length):
    """Make initial population.

    Args:
      pop_size: number of genotypes in population.
      evaluator: An Evaluator class instance for generating initial genotype.
      image_size: Size of initial image for genotype to draw on.
      input_spec_size: Sequence element size
      sequence_length: Initial length of sequences

    Returns:
      Initialised population.
    """
    print(f"Creating initial population of size {pop_size}")
    pop = []
    for _ in range(pop_size):
        a_genotype = evaluator.make_initial_genotype(
            img=np.zeros((image_size, image_size, 3), dtype=np.uint8),
            sequence_length=sequence_length,
            input_spec_size=input_spec_size)
        pop.append(a_genotype)
    return pop

def evolution_loop(population, worker_pool, evaluator, num_generations,
                   use_crossover, crossover_prob,
                   use_projective_transform, projective_transform_coefficient,
                   plot_every, plot_batch):
    """Create population and run evolution.

    Args:
      population: Initial population of genotypes
      worker_pool: List of workers of parallel evaluations
      evaluator: image evaluator to calculate fitnesses
      num_generations: number of generations to run
      use_crossover: Whether crossover is used for offspring
      crossover_prob: Probability that crossover takes place
      use_projective_transform: Use projective transforms in evaluation
      projective_transform_coefficient: Degree of projective transform
      plot_every: number of generations between new plots
      plot_batch: whether to show all samples in the batch then plotting
    """
    population_size = len(population)
    num_workers = len(worker_pool)
    print("Population of {} genotypes being evaluated by {} workers".format(
        population_size, num_workers))
    drawings = {}
    fitness_history = []
    init_gen = len(fitness_history)
    print(f"(Re)starting evolution at generation {init_gen}")
    for gen in range(init_gen, num_generations):

        # Drawing
        t0_loop = time.time()
        futures = []
        for j in range(0, population_size, num_workers):
            for i in range(num_workers):
                futures.append(worker_pool[i].compute.remote(
                    cloudpickle.dumps(population[i+j]), i+j))
            data = ray.get(futures)
            for i in range(num_workers):
                drawings[data[i+j]["id"]] = data[j+i]["drawing"]
        batch_drawings = []
        for i in range(population_size):
            batch_drawings.append(drawings[i])

        # Fitness evaluation using CLIP
        t1_loop = time.time()
        fitnesses = get_fitness(batch_drawings, use_projective_transform,
                                projective_transform_coefficient)
        fitness_history.append(copy.deepcopy(fitnesses))

        # Tournament
        t2_loop = time.time()
        population, best_genotype = truncation_selection(
            population, fitnesses, evaluator, use_crossover, crossover_prob)
        t3_loop = time.time()
        duration_draw = t1_loop - t0_loop
        duration_fit = t2_loop - t1_loop
        duration_tournament = t3_loop - t2_loop
        duration_total = t3_loop - t0_loop
        if gen % plot_every == 0:
            if VISUALIZE_GENOTYPE:
                evaluator.drawing_generator.visualize_genotype(best_genotype)
            print("Draw: {:.2f}s fit: {:.2f}s evol: {:.2f}s total: {:.2f}s".format(
                duration_draw, duration_fit, duration_tournament, duration_total))
            plot_training_res(batch_drawings, fitness_history)
            if plot_batch:
                num_samples_to_plot = int(math.pow(
                    math.floor(np.sqrt(population_size)), 2))
                plot_samples(batch_drawings, num_samples=num_samples_to_plot)


# # Configure and Generate

# In[ ]:


#@title Hyperparameters

#@markdown Evolution parameters: population size and number of generations.
POPULATION_SIZE = 10  #@param {type:"slider", min:4, max:100, step:2}
NUM_GENERATIONS = 5000  #@param {type:"integer", min:100}
#@markdown Number of workers working in parallel (should be equal to or smaller than the population size).
NUM_WORKERS = 10  #@param {type:"slider", min:4, max:100, step:2}
#@markdown Crossover in evolution.
USE_CROSSOVER = True  #@param {type:"boolean"}
CROSSOVER_PROB = 0.01  #@param {type:"number"}
#@markdown Number of LSTMs, each one encoding a group of strokes.
NUM_LSTMS = 5  #@param {type:"integer", min:1, max:5}
#@markdown Number of inputs for each element in the input sequences.
INPUT_SPEC_SIZE = 10  #@param {type:"integer"}
#@markdown Length of the input sequence fed to the LSTMs (determines number of strokes).
SEQ_LENGTH = 20  #@param {type:"integer", min:20, max:200}
#@markdown Rendering parameter.
ALLOW_STROKES_BEYOND_IMAGE_EDGES = True  #@param {type:"boolean"}
#@markdown CLIP evaluation: do we use projective transforms of images?
USE_PROJECTIVE_TRANSFORM = True  #@param {type:"boolean"}
PROJECTIVE_TRANSFORM_COEFFICIENT = 0.000001  #@param {type:"number"}
#@markdown These parameters should be edited mostly only for debugging reasons.
NET_LSTM_HIDDENS = 40  #@param {type:"integer"}
NET_MLP_HIDDENS = 20  #@param {type:"integer"}
# Scales the values used in genotype's initialisation.
DNA_SCALE = 1.0  #@param {type:"number"}
IMAGE_SIZE = 224  #@param {type:"integer"}
VERBOSE_CODE = False  #@param {type:"boolean"}
VISUALIZE_GENOTYPE = False  #@param {type:"boolean"}
VERBOSE_MUTATION = False  #@param {type:"boolean"}
#@markdown Number of generations between new plots.
PLOT_EVERY_NUM_GENS = 5  #@param {type:"integer"}
#@markdown Whether to show all samples in the batch when plotting.
PLOT_BATCH = True  # @param {type:"boolean"}

assert POPULATION_SIZE % NUM_WORKERS == 0, "POPULATION_SIZE not multiple of NUM_WORKERS"


# #Running the original evolutionary algorithm
# This is the original inefficient version of Arnheim which uses a genetic algorithm to optimize the picture. It takes at least 12 hours to produce an image using 50 workers. In our paper we used 500-1000 GPUs which speeded things up considerably. Refer to Arnheim 2 for a far more efficient way to generate images with a similar architecture.
#
# Try prompts like “A photorealistic chicken”. Feel free to modify this colab to include your own way of generating and evolving images like we did in figure 2 here https://arxiv.org/pdf/2105.00162.pdf.

# In[ ]:


# @title Get text input and run evolution
PROMPT = "an apple"  #@param {type:"string"}

# Tokenize prompts and coompute CLIP features.
text_input = clip.tokenize(PROMPT).to(device)
with torch.no_grad():
    text_features = model.encode_text(text_input)

ray.shutdown()
ray.init()

drawing_lstm_arch = DrawingLSTMSpec(INPUT_SPEC_SIZE,
                                    NUM_LSTMS,
                                    NET_LSTM_HIDDENS,
                                    NET_MLP_HIDDENS)

workers = create_workers(NUM_WORKERS, IMAGE_SIZE, drawing_lstm_arch,
                         ALLOW_STROKES_BEYOND_IMAGE_EDGES)


drawing_evaluator = Evaluator(IMAGE_SIZE, drawing_lstm_arch,
                              ALLOW_STROKES_BEYOND_IMAGE_EDGES)

drawing_population = make_population(POPULATION_SIZE, drawing_evaluator,
                                     IMAGE_SIZE, INPUT_SPEC_SIZE, SEQ_LENGTH)

evolution_loop(drawing_population, workers, drawing_evaluator, NUM_GENERATIONS,
               USE_CROSSOVER, CROSSOVER_PROB,
               USE_PROJECTIVE_TRANSFORM, PROJECTIVE_TRANSFORM_COEFFICIENT,
               PLOT_EVERY_NUM_GENS, PLOT_BATCH)
