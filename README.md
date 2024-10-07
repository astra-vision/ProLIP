# Fine-tuning CLIP's Last Visual Projector: A Few-Shot Cornucopia
[Mohammad Fahes<sup>1</sup>](https://mfahes.github.io/),
[Tuan-Hung Vu<sup>1,2</sup>](https://tuanhungvu.github.io/),
[Andrei Bursuc<sup>1,2</sup>](https://abursuc.github.io/),
[Patrick Pérez<sup>3</sup>](https://ptrckprz.github.io/),
[Raoul de Charette<sup>1</sup>](https://team.inria.fr/rits/membres/raoul-de-charette/)</br>
<sup>1</sup> Inria, Paris, France.

<sup>2</sup> valeo.ai, Paris, France.

<sup>3</sup> Kyutai, Paris, France.<br>

TL; DR: CLIP projects visual embedding to the shared latent space using a linear projection layer. We show that simply fine-tuning this guy (:p) can be a strong alternative to linear probing, prompt tuning and CLIP-adapters, and performs also well on test-time adaptation.

Stay tuned for the code!


## ProLIP
<p align="center">
  <b>We fine-tune the pretrained linear projection layer of the vision encoder with a regularization loss towards the pre-trained weights.</b>
</p>
<p align="center">
  <img src="./figures/prolip_figure.png/" style="width:100%"/>
</p>