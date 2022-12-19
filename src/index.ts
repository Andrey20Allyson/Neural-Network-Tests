import { NeuralNetwork, NeuronPattern } from "./NeuralNetwork";

type dataType = [number, number];

const pattern = new NeuronPattern(NeuronPattern.SUM_NF, NeuronPattern.STEP_AF);
const network = new NeuralNetwork(2, 1);

console.log('Criando inteligencia...');

let column = network.newColumn();
network.push(column, pattern, 2);

column = network.newColumn();
network.push(column, pattern, 1);

network.initWeights();

console.log('Inteligencia criada!');

const data: dataType[] = [
    [1, 0],
    [0, 1],
    [0, 1],
    [1, 1]
];

const resp = [
    [1],
    [0],
    [0],
    [0]
];

console.log('Inteligencia está sendo treinada...');

const learnInit = Date.now();
network.learn(1, 4000, data, resp);

console.log('Inteligencia foi treinada em %sms!', Date.now() - learnInit);

console.log('Executando inteligencia...\n');

for (let d of data) {
    const result = network.exec(d);

    console.log(Math.round(result[0]));
}
