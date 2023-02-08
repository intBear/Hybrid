import torch
import tqdm
from collections import OrderedDict
from utils import *
from torchvision.utils import save_image


class LearningRateSchedule:
    def get_learning_rate(self, epoch):
        pass

class StepLearningRateSchedule(LearningRateSchedule):
    def __init__(self, initial, interval, factor):
        self.initial = initial
        self.interval = interval
        self.factor = factor

    def get_learning_rate(self, epoch):

        return self.initial * (self.factor ** (epoch // self.interval))

def get_learning_rate_schedules(args):
    schedules = []
    if args.lr_type == "Step":
        schedules.append(StepLearningRateSchedule(0.0005, 500, 0.5))


class Trainer():
    def __init__(
            self,
            device,
            image,
            decoder,
            rd_losses,
            lat_layer,
            args,
            print_freq=1,
    ):
        """Model to learn a representation of a single datapoint.

        Args:
            representation (siren.Siren): Neural net representation of image to
                be trained.
            lr (float): Learning rate to be used in Adam optimizer.
            print_freq (int): Frequency with which to print losses.
        """
        self.device = device
        self.image = image
        self.decoder = decoder
        self.args = args
        self.optimizer_all = torch.optim.Adam([
            {
                "params": decoder.parameters(),
                "lr": args.lr0,
            },
            {
                "params": lat_layer.parameters(),
                "lr": args.lr1,
            },
        ])
        self.rd_losses = rd_losses
        self.lat_layer = lat_layer
        if self.args.resume:
            path_checkpoint = "./models/checkpoint/L1_check.pth"
            checkpoint = torch.load(path_checkpoint)
            self.decoder.load_state_dict(checkpoint['net'])
            self.optimizer_all.load_state_dict(checkpoint['optimizer'])
        self.model_path = args.model_path
        self.print_freq = print_freq
        self.steps = 0  # Number of steps taken in training
        self.loss_func = torch.nn.MSELoss()
        self.best_vals = {'psnr': 0.0, 'loss': 1e8}
        self.logs = {'psnr': [], 'loss': [], 'bpp': []}
        # self.logs = {'psnr': [], 'loss': []}
        # Store parameters of best model (in terms of highest PSNR achieved)
        # self.best_model = OrderedDict((k, v.detach().clone()) for k, v in self.decoder.state_dict().items())

    def train(self, num_iters):
        """Fit neural net to image.

        Args:
            coordinates (torch.Tensor): Tensor of coordinates.
                Shape (num_points, coordinate_dim).
            features (torch.Tensor): Tensor of features. Shape (num_points, feature_dim).
            num_iters (int): Number of iterations to train for.
        """
        C, H, W = self.image.shape
        cords, features = to_CordsAndValues(self.image, self.device)
        embedder, out_dim = get_positional_embedder(4, True)
        embedder = embedder.to(self.device)
        p_inf = embedder(cords)
        p_inf = p_inf.reshape(H, W, out_dim)
        indices = torch.LongTensor([0]).to(self.device)
        with tqdm.trange(num_iters, ncols=250) as t:
            for i in t:
                self.optimizer_all.zero_grad()
                latents = self.lat_layer(indices).squeeze()
                y, z = self.decoder.get_features(latents, self.image)
                z = torch.cat((z, p_inf), 2)
                # Update model
                predicted = self.decoder(z)
                # bpp = self.decoder.cul_bpp(y)
                # loss = self.rd_losses.output(predicted, features, bpp)
                loss = self.rd_losses.output(predicted, features)
                loss["psnr"] = mse2psnr(loss["mse_loss"])
                loss["loss"] = loss["mse_loss"]
                loss["mse_loss"].backward()
                self.optimizer_all.step()


                # Print results and update logs
                log_dict = {'loss': loss["loss"],
                            'psnr': loss["psnr"],
                            # 'bpp' : loss["bpp_loss"],
                            'best_psnr': self.best_vals['psnr']
                            }
                t.set_postfix(**log_dict)
                # for key in ['loss', 'psnr', 'bpp']:
                for key in ['loss', 'psnr']:
                    self.logs[key].append(log_dict[key])

                # Update best values
                if loss["loss"] < self.best_vals['loss']:
                    self.best_vals['loss'] = loss["loss"]
                if loss["psnr"] > self.best_vals['psnr']:
                    self.best_vals['psnr'] = loss["psnr"]
                    # If model achieves best PSNR seen during training, update model
                    if i > int(num_iters / 2.):
                        checkpoint = {
                            "net": self.decoder.state_dict(),
                            "optimizer": self.optimizer_all.state_dict(),
                            "epoch": i
                        }
                        if not os.path.isdir("./models/checkpoint"):
                            os.mkdir("./models/checkpoint")
                        torch.save(checkpoint, './models/checkpoint/L1_check.pth')
                        # save predicted image
                if i > int(num_iters - 2):
                    with torch.no_grad():
                        img_recon = predicted.reshape(C, H, W)
                        save_image(torch.clamp(img_recon, 0, 1).to('cpu'),
                                   self.args.logdir + f'/18_reconstruction_{i}.png')

