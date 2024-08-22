import torch


class Logger(object):
    def __init__(self, runs, info=None):
        self.info = info
        self.results = [[] for _ in range(runs)]

    def add_result(self, run, result):
        assert len(result) == 3
        assert run >= 0 and run < len(self.results)
        self.results[run].append(result)
        
    # def print_statistics(self, run=None):
    #     if run is not None:
    #         result = 100 * torch.tensor(self.results[run])
    #         argmax = result[:, 1].argmax().item()
    #         print(f'Run {run + 1:02d}:')
    #         print(f'Highest Train: {result[:, 0].max():.2f}')
    #         print(f'Highest Valid: {result[:, 1].max():.2f}')
    #         print(f'  Final Train: {result[argmax, 0]:.2f}')
    #         print(f'   Final Test: {result[argmax, 2]:.2f}')
    #     else:
    #         result = 100 * torch.tensor(self.results)

    #         best_results = []
    #         for r in result:
    #             train1 = r[:, 0].max().item()
    #             valid = r[:, 1].max().item()
    #             train2 = r[r[:, 1].argmax(), 0].item()
    #             test = r[r[:, 1].argmax(), 2].item()
    #             best_results.append((train1, valid, train2, test))

    #         best_result = torch.tensor(best_results)

    #         print(f'All runs:')
    #         r = best_result[:, 0]
    #         print(f'Highest Train: {r.mean():.2f} ± {r.std():.2f}')
    #         r = best_result[:, 1]
    #         print(f'Highest Valid: {r.mean():.2f} ± {r.std():.2f}')
    #         r = best_result[:, 2]
    #         print(f'  Final Train: {r.mean():.2f} ± {r.std():.2f}')
    #         r = best_result[:, 3]
    #         print(f'   Final Test: {r.mean():.2f} ± {r.std():.2f}')

    # 改過的
    def print_statistics(self, run=None):
        if run is not None:
            results = self.results[run]
            if len(results) == 0:
                results = [[0, 0, 0]]  # Default value if no data
            
            result = 100 * torch.tensor(results).float()
            if result.dim() == 1:
                result = result.unsqueeze(0)  # Make it 2D if it's not
            
            argmax = result[:, 1].argmax().item() if result.size(0) > 1 else 0
            
            print(f'Run {run + 1:02d}:')
            print(f'Highest Train: {result[:, 0].max():.2f}')
            print(f'Highest Valid: {result[:, 1].max():.2f}')
            print(f'  Final Train: {result[argmax, 0]:.2f}')
            print(f'   Final Test: {result[argmax, 2]:.2f}')
        else:
            # result = 100 * torch.tensor(self.results)
            
            # Handle the tensor conversion properly
            # all_results = [torch.tensor(res).unsqueeze(0) if torch.tensor(res).dim() == 1 else torch.tensor(res) for res in self.results if len(res) > 0]
            # result = torch.cat(all_results, dim=0)
            
            # best_results = []
            # for r in result:
            #     train1 = r[:, 0].max().item()
            #     valid = r[:, 1].max().item()
            #     train2 = r[r[:, 1].argmax(), 0].item()
            #     test = r[r[:, 1].argmax(), 2].item()
            #     best_results.append((train1, valid, train2, test))

            best_results = []
            for res in self.results:
                if len(res) > 0:
                    tensor_res = torch.tensor(res).float()
                    if tensor_res.dim() == 1:
                        tensor_res = tensor_res.unsqueeze(0)
                else:
                    tensor_res = torch.tensor([[0, 0, 0]]).float()  # Default value

                train1 = tensor_res[:, 0].max().item()
                valid = tensor_res[:, 1].max().item()
                train2 = tensor_res[tensor_res[:, 1].argmax(), 0].item() if tensor_res.size(0) > 1 else 0
                test = tensor_res[tensor_res[:, 1].argmax(), 2].item() if tensor_res.size(0) > 1 else 0
                best_results.append((train1, valid, train2, test))

            if not best_results:
                best_results = [[0, 0, 0, 0]]  # Default value if completely empty
                
            best_result = torch.tensor(best_results).float()

            print(f'All runs:')
            r = best_result[:, 0]
            print(f'Highest Train: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 1]
            print(f'Highest Valid: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 2]
            print(f'  Final Train: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 3]
            print(f'   Final Test: {r.mean():.2f} ± {r.std():.2f}')
