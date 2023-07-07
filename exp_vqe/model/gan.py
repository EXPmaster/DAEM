import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):

    def __init__(self, num_miti, num_qubits=4, hidden_size=128):
        super().__init__()
        self.obs_embed = nn.Linear(8 * 2 * 2, hidden_size)
        self.param_embed = nn.Linear(1, hidden_size)
        self.scale_embed = nn.Linear(1, hidden_size)
        self.noise_embed = nn.Linear(64, hidden_size)
        self.embed_ln = nn.LayerNorm(hidden_size)
        # self.net = nn.Sequential(
        #     nn.Linear(5 * hidden_size, 256),
        #     nn.Mish(inplace=True),
        #     nn.Linear(256, 512),
        #     nn.Mish(inplace=True),
        #     nn.Linear(512, 4 * num_qubits)
        # )
        self.net = nn.Sequential(
            nn.Linear(4 * hidden_size, 512),
            # nn.Dropout(0.5),
            nn.Mish(inplace=True),
            nn.Linear(512, 1024),
            # nn.Dropout(0.5),
            nn.Mish(inplace=True),
            nn.Linear(1024, 1024),
            nn.Mish(inplace=True),
            # nn.Linear(1024, 4 ** num_miti),
            nn.Linear(1024, 1),
            # nn.Tanh()
        )

    def forward(self, noise, params, obs, pos, scale):
        obs_flat = torch.cat((obs.real, obs.imag), -1).flatten(1)
        obs_ebd = self.obs_embed(obs_flat)
        
        noise = self.noise_embed(noise)
        param_ebd = self.param_embed(params)
        scale_ebd = self.scale_embed(scale)
        x = torch.cat((noise, param_ebd, obs_ebd, scale_ebd), 1)
        # x = torch.cat((noise.unsqueeze(1), obs_ebd, scale_ebd.unsqueeze(1)), 1)
        # x = self.embed_ln(x).flatten(1)
        x = self.net(x)
        # x = x.view(-1, self.num_qubits, 4)
        # x = torch.softmax(x, -1)
        return x
    
    def load_envs(self, args, force=False):
        if os.path.exists('tmp.pt') and not force:
            print('Loading from buffer...')
            buffer = torch.load('tmp.pt', map_location='cpu')
            self.register_buffer('rhos', buffer['rhos'])
        else:
            rhos = {}
            print('Loading envs...')
            for env_name in tqdm(os.listdir(args.env_path)):
                param = float(env_name.replace('.pkl', '').split('_')[-1])
                env_path = os.path.join(args.env_path, env_name)
                env = IBMQEnv.load(env_path)
                num_qubits = env.circuit.num_qubits
                rho = []
                for i in range(num_qubits - 1):
                    reduced_rho = []
                    for r in env.state_table:
                        reduced_rho.append(partial_trace(r, [i, i+1], [2] * num_qubits))
                    reduced_rho = np.array(reduced_rho)
                    rho.append(reduced_rho)
                rhos[param] = rho
            buffer = np.array([v for _, v in sorted(rhos.items(), key=lambda x: x[0])])
            self.register_buffer('rhos', torch.tensor(buffer, dtype=torch.cfloat))
            if not force:
                torch.save(self.state_dict(), 'tmp.pt')
    
    def expectation_from_prs(self, params, observables, positions, quasi_probs):
        """
        Calculate expectation from quasi-probabilities.

        Args:
            params: (int) converted coeffs of circuits.
            observables: (torch.tensor[bs, 4, 4])
            positions: (torch.tensor[bs, 2]) qubit position of observables.
            quasi_probs: (torch.tensor[bs, 4096])

        Returns:
            Expectation [bs, 1]
        """
        rho = self.rhos[params.flatten()]  # .squeeze(1)
        rho = rho[torch.arange(params.shape[0]), positions[:, 0]]
        meas_results = torch.matmul(rho, observables.unsqueeze(1)).diagonal(dim1=-2, dim2=-1).sum(-1).real
        return (quasi_probs * meas_results).sum(1, keepdim=True)


class Discriminator(nn.Module):

    def __init__(self, num_miti, num_qubits=4, hidden_size=128):
        super().__init__()
        self.num_miti = num_miti
        self.obs_embed = nn.Linear(8 * 2 * 2, hidden_size)
        self.meas_embed = nn.Linear(1, hidden_size)
        # self.meas_embed = nn.Sequential(
        #     nn.Linear(2 ** num_miti + 1, 256),
        #     # nn.Dropout(0.2),
        #     nn.LeakyReLU(),
        #     nn.Linear(256, hidden_size),
        # )
        # self.meas_cut = nn.Linear(2 ** num_miti + 1, hidden_size)
        
        self.param_embed = nn.Linear(1, hidden_size)
        self.embed_ln = nn.LayerNorm(hidden_size)
        self.scale_embed = nn.Linear(1, hidden_size)
        self.discriminator = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 1),
            # nn.Sigmoid()
        )
        self.backbone = nn.Sequential(
            nn.Linear(2 * hidden_size, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256)
        )
        # self.net1 = nn.Sequential(
        #     nn.Linear(256, 128),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Linear(128, 64),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Linear(64, 1)
        # )
        # self.net2 = nn.Sequential(
        #     nn.Linear(256, 128),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Linear(128, 64),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Linear(64, 1)
        # )
        # self.net3 = nn.Sequential(
        #     nn.Linear(256, 128),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Linear(128, 64),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Linear(64, 1)
        # )

    def forward(self, meas_in, params, obs, pos, scale):
        obs_flat = torch.cat((obs.real, obs.imag), -1).flatten(1)
        obs_ebd = self.obs_embed(obs_flat)

        meas_ebd = self.meas_embed(meas_in)
        # meas_shortcut = self.meas_cut(meas_in)
        # meas_ebd = meas_ebd + meas_shortcut

        param_ebd = self.param_embed(params)
        scale_ebd = self.scale_embed(scale)
        # x = torch.cat((param_ebd.unsqueeze(1), obs_ebd, scale_ebd.unsqueeze(1)), 1).flatten()
        x = torch.cat((param_ebd, obs_ebd), 1)
        x = self.backbone(x)
        # coeff1 = self.net1(x)
        # coeff2 = self.net2(x)
        # coeff3 = self.net3(x)
        # # pred_noisy = coeff2 * torch.exp(coeff1 * scale) + coeff3
        # pred_noisy = coeff1 + coeff2 * scale + coeff3 * scale ** 2
        cat_input = torch.cat((x, meas_ebd, scale_ebd), 1)
        output_disc = self.discriminator(cat_input)
        # pred_noisy = self.net2(x) * torch.exp(self.net1(x) * scale) + self.net3(x)
        return output_disc  # , (pred_noisy - meas_in) ** 2
