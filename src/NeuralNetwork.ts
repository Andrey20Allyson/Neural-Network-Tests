export type NeuronFunction = (I: number[], W: number[]) => number;
export type ActivationFunction = (value: number) => number;
export type Weights = number[][][];

export class NeuronPattern {
    static readonly SUM_NF: NeuronFunction = (I, W) => {
        const minLen = I.length < W.length? I.length: W.length;
        let result = 0;

        for (let i = 0; i < minLen; i++)
            result += I[i] * W[i];

        return result;
    };

    static readonly STEP_AF: ActivationFunction = (value) => {
        return value > 0? value: 0;
    }

    static readonly SIGM_AF: ActivationFunction = (value) => {
        return Math.tanh(value);
    };

    private readonly neuronFunction: NeuronFunction;
    private readonly activationFunction: ActivationFunction;

    constructor(neuronF: NeuronFunction, activationF: ActivationFunction) {
        this.neuronFunction = neuronF;
        this.activationFunction = activationF;
    }

    exec(I: number[], W: number[]) {
        return this.activationFunction(this.neuronFunction(I, W));
    }
}

export class NeuralNetwork {
    network: NeuronPattern[][];
    weights: Weights;

    readonly numberOfInputs: number;
    readonly numberOfOutputs: number;

    constructor(nInputs: number = 1, nOutputs: number = 1) {
        this.network = [];
        this.weights = [];

        this.numberOfInputs = nInputs;
        this.numberOfOutputs = nOutputs;
    }

    newColumn() {
        const newColumnIndex = this.network.length;
        this.network[newColumnIndex] = [];
        return newColumnIndex;
    }

    push(columnIndex: number, neuronPattern: NeuronPattern, quantity: number) {
        if (this.network[columnIndex] === undefined) 
            throw new Error('out of bounds');

        for (let i = 0; i < quantity; i++)
            this.network[columnIndex].push(neuronPattern);
    }

    initWeights(initialValue: number = 0) {
        this.weights[0] = [];

        for (let i = 0; i < this.network[0].length; i++) {
            this.weights[0][i] = [];

            for (let j = 0; j < this.numberOfInputs; j++) {
                this.weights[0][i][j] = initialValue;
            }
        }

        for (let i = 1; i < this.network.length; i++) {
            this.weights[i] = [];

            for (let j = 0; j < this.network[i].length; j++) {
                this.weights[i][j] = [];

                for (let k = 0; k < this.network[i - 1].length; k++) {
                    this.weights[i][j][k] = initialValue;
                }
            }
        }
    }

    private rawExec(inputs: number[]) {
        let results: number[][] = [[], []];

        

        for (let i = 0; i < inputs.length; i++)
            results[0][i] = inputs[i];

        for (let i = 0; i < this.network.length; i++) {
            for (let j = 0; j < this.network[i].length; j++)
                results[1][j] = this.network[i][j].exec(results[0], this.weights[i][j]);

            results[0] = results[1];
            results[1] = [];
        }

        return results[0];
    }
    
    exec(inputs: number[]) {
        if (!this.canExec(inputs)) throw new Error('cant execute');

        return this.rawExec(inputs);
    }

    canExec(inputs: number[]): boolean {
        return !(inputs.length !== this.numberOfInputs || this.network[this.network.length - 1].length !== this.numberOfOutputs);
    }

    cloneWeigths() {
        const weights: Weights = [];

        for (let i = 0; i < this.weights.length; i++) {
            weights[i] = [];
            
            for (let j = 0; j < this.weights[i].length; j++) {
                weights[i][j] = [];

                for (let k = 0; k < this.weights[i][j].length; k++) {
                    weights[i][j][k] = this.weights[i][j][k];
                }
            }
        }

        return weights;
    }

    learn(times: number, population: number, inputs: number[][], resp: number[][]) {
        if (!this.canExec(inputs[0] ?? [])) throw new Error('cant execute');

        for (let i = 0; i < times; i++) {
            let individuals: Weights[] = [];

            let minDif = Number.MAX_VALUE;
            let bestIndividual: Weights = this.weights;

            for (let j = 0; j < population; j++) {
                let weight = this.cloneWeigths();
                NeuralNetwork.randomizeWeights(2 / (i + 1), weight);

                individuals[j] = weight;
            }

            for (let j = 0; j < individuals.length; j++) {
                this.weights = individuals[j];

                let dif = 0;  

                for (let k = 0; k < inputs.length; k++) {
                    let result = this.rawExec(inputs[k]);

                    for (let t = 0; t < result.length; t++) {
                        dif += Math.abs(result[t] - resp[k][t]);
                    }
                }

                if (dif < minDif) {
                    bestIndividual = individuals[j];
                    minDif = dif;
                }
            }

            this.weights = bestIndividual;
        }
    }

    static randomizeWeights(multiplier: number, weights: Weights): void {
        for (let i = 0; i < weights.length; i++) {
            for (let j = 0; j < weights[i].length; j++) {
                for (let k = 0; k < weights[i][j].length; k++) {
                    const r = ((Math.random() * 2) - 1) * multiplier;
                    weights[i][j][k] += r;
                }
            }
        }
    }
}