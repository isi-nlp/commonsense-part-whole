import numpy as np
import visdom

#define a plotter object to make it simple to carry many window objects around
class Plotter:
    def __init__(self, args):
        import visdom
        self.vis = visdom.Visdom(env="%s" % args.exec_time)
        self.batch_loss_plt = self.vis.line(np.array([[0, 0]]), np.array([[0, 0]]),
                                            opts={'title': "Batch loss",
                                                "xlabel": "Batch",
                                                "legend": ["loss", "moving average"],
                                                "showlegend": True})
        self.show_hps(args.__dict__)

    def populate(self, metrics_dv, metrics_tr):
        x = np.array([0])

        self.dv_prec_plt = self.vis.line(metrics_dv['prec'][-1:], x,
                            opts={'title': "Dev precision", "xlabel": "Epoch"})
        self.dv_rec_plt = self.vis.line(metrics_dv['rec'][-1:], x,
                            opts={'title': "Dev recall", "xlabel": "Epoch"})
        self.dv_f1_plt = self.vis.line(metrics_dv['f1'][-1:], x,
                            opts={'title': "Dev F1", "xlabel": "Epoch"})
        self.dv_mse_plt = self.vis.line(metrics_dv['mse'][-1:], x,
                            opts={'title': "Dev MSE", "xlabel": "Epoch"})
        self.dv_rho_plt = self.vis.line(metrics_dv['spearman'][-1:], x,
                            opts={'title': "Dev spearman's rho", "xlabel": "Epoch"})
        self.tr_loss_plt = self.vis.line(metrics_tr['loss'][-1:], x,
                            opts={'title': "Train loss", "xlabel": "Epoch"})
        self.dv_loss_plt = self.vis.line(metrics_dv['loss_dev'][-1:], x,
                            opts={'title': "Dev loss", "xlabel": "Epoch"})

    def update(self, epoch, metrics_dv, metrics_tr):
        x = np.array([epoch])

        self.vis.line(metrics_dv['prec'][-1:], x, win=self.dv_prec_plt, update='append')
        self.vis.line(metrics_dv['rec'][-1:], x, win=self.dv_rec_plt, update='append')
        self.vis.line(metrics_dv['f1'][-1:], x, win=self.dv_f1_plt, update='append')
        self.vis.line(metrics_dv['mse'][-1:], x, win=self.dv_mse_plt, update='append')
        self.vis.line(metrics_dv['spearman'][-1:], x, win=self.dv_rho_plt, update='append')
        self.vis.line(metrics_dv['loss_dev'][-1:], x, win=self.dv_loss_plt, update='append')
        self.vis.line(metrics_tr['loss'][-1:], x, win=self.tr_loss_plt, update='append')

    def plot_batch_loss(self, losses, avg_run):
        x = np.arange(len(losses))
        l_avg = [np.mean(losses[max(i-avg_run,0):i+1]) for i in range(len(losses))]
        data = np.vstack([losses, l_avg]).transpose()
        self.vis.line(data, x, win=self.batch_loss_plt, update='new')

    def show_hps(self, params):
        self.text = self.vis.text("All hyperparameters:")
        for key, val in params.items():
            self.vis.text("%s: %s" % (str(key), str(val)), win=self.text, append=True)
