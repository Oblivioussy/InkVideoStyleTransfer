import time
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
import pickle, os

opt = TrainOptions().parse()
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('#training images = %d' % dataset_size)

while True:
    total_steps = opt.total_steps
    old_lambda_sup = opt.lambda_sup
    lambda_sup = old_lambda_sup
    model = create_model(opt)
    visualizer = Visualizer(opt)
    try:
        for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
            epoch_start_time = time.time()
            epoch_iter = 0

            for i, data in enumerate(dataset):
                print("total_steps:", total_steps)
                iter_start_time = time.time()
                visualizer.reset()
                total_steps += opt.batchSize
                epoch_iter += opt.batchSize
                model.set_input(data)
                if epoch >= opt.start_dec_sup:
                    lambda_sup = (0.3+0.7*(opt.niter + opt.niter_decay-epoch) /
                                (opt.niter + opt.niter_decay-opt.start_dec_sup))*old_lambda_sup
                else:
                    lambda_sup = old_lambda_sup
                
                # print(epoch, opt.start_add_loss)
                if epoch >= opt.start_add_loss:
                    if epoch >= opt.end_add_loss:
                        lambda_newloss = 1
                    else:
                        lambda_newloss = (epoch-opt.start_add_loss)/(opt.end_add_loss-opt.start_add_loss)
                else:
                    lambda_newloss = 0
                
                model.optimize_parameters(lambda_sup, lambda_newloss, epoch=epoch)

                if total_steps % opt.display_freq == 0:
                    save_result = total_steps % opt.update_html_freq == 0
                    visualizer.display_current_results(
                        model.get_current_visuals(), epoch, save_result)

                if total_steps % opt.print_freq == 0:
                    errors = model.get_current_errors()
                    t = (time.time() - iter_start_time) / opt.batchSize
                    visualizer.print_current_errors(epoch, epoch_iter, errors, t, total_steps)
                    visualizer.print_display_param(model.display_param, total_steps)
                    # if opt.display_id > 0:
                    #     visualizer.plot_current_errors(epoch, float(
                    #         epoch_iter)/dataset_size, opt, errors)

                if total_steps % opt.save_latest_freq == 0:
                    print('saving the latest model (epoch %d, total_steps %d)' %
                        (epoch, total_steps))
                    model.save('latest')
                    opt.epoch_count = epoch + 1
                    opt.which_epoch = 'latest'
                # exit(0)
                

            if epoch % opt.save_epoch_freq == 0:
                print('saving the model at the end of epoch %d, iters %d' %
                    (epoch, total_steps))
                model.save('latest')
                model.save(epoch)
                opt.epoch_count = epoch

            print('End of epoch %d / %d \t Time Taken: %d sec' %
                (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
            model.update_learning_rate()
        break
    except Exception as e:
        raise e
        pass
