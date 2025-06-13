
export async function trainModelFromFile(inputId) {
    const file = document.getElementById(inputId).files?.[0];
    if (!file) {
        alert("Upload een JSON-bestand.");
        return;
    }

    let allData;
    try {
        allData = JSON.parse(await file.text());
    } catch {
        alert("Ongeldig JSON-bestand.");
        return;
    }

    // Stap 1: Schud en splits de data (80% training, 20% test)
    allData.sort(() => Math.random() - 0.5); // Willekeurig husselen

    const splitPoint = Math.floor(allData.length * 0.8);
    const trainingData = allData.slice(0, splitPoint);
    const testData = allData.slice(splitPoint);

    console.log(`Totaal ${allData.length} voorbeelden geladen.`);
    console.log(`--> ${trainingData.length} voor training.`);
    console.log(`--> ${testData.length} voor testen.`);

    if (testData.length === 0) {
        alert("Te weinig data om een testset te maken. Verzamel meer voorbeelden.");
        return;
    }

    // Stap 2: Maak het model en voeg alleen de trainingsdata toe
    const model = ml5.neuralNetwork({
        task: 'classification',
        debug: true,
        inputs: 66,
        outputs: ['Squat', 'JumpingJack'],
        learningRate: 0.01
    });

    for (const item of trainingData) {
        model.addData(item.keypoints, { label: item.label });
    }

    // Stap 3: Train het model
    console.log("Starten met trainen...");
    await new Promise((resolve, reject) => {
        model.train({ epochs: 50, batchSize: 32 }, (err) => err ? reject(err) : resolve());
    });
    console.log("Training voltooid.");

    // Stap 4: Evalueer het model met de testdata
    console.log("Model evalueren met de testdata...");
    let correctPredictions = 0;
    for (const testItem of testData) {
        // Gebruik een Promise om de asynchrone 'classify' aanroep af te wachten
        const results = await new Promise(resolve => {
            model.classify(testItem.keypoints, (err, res) => resolve(res));
        });

        if (results[0].label === testItem.label) {
            correctPredictions++;
        }
    }

    // Stap 5: Bereken en toon de accuracy
    const accuracy = (correctPredictions / testData.length) * 100;
    alert(`Evaluatie voltooid!\n\nAccuracy op de testset: ${accuracy.toFixed(2)}%\n(${correctPredictions} van de ${testData.length} correct voorspeld)`);

    // Geef het volledig getrainde model terug
    return model;
}


// --- DE REST VAN JE CODE BLIJFT HETZELFDE ---

export async function saveModel(model) {
    model.save('poseModel'); // download model.json + model.weights.bin
}

export async function trainModel(data) {
    const model = ml5.neuralNetwork({
        task: 'classification',
        debug: true,
        inputs: 66,
        outputs: 2,
        learningRate: 0.01,
    });

    data.forEach(sample => {
        model.addData(sample.keypoints, { label: sample.label });
    });

    await new Promise((resolve, reject) => {
        model.train({
            epochs: 50,
            batchSize: 32,
            callback: (epoch, loss) => {
                console.log(`Epoch ${epoch} - Loss: ${loss}`);
            }
        }, (err) => {
            if (err) {
                reject(err);
            } else {
                resolve();
            }
        });
    });

    return model;
}

function flattenKeypoints(keypoints) {
    if (!keypoints || keypoints.length === 0) {
        console.error("Ongeldige keypoints:", keypoints);
        return [];
    }
    return keypoints.reduce((acc, value, index) => {
        if (index % 2 === 0) {
            acc.push(value);
        } else if (index % 2 === 1) {
            acc.push(value);
        }
        return acc;
    }, []);
}

function oneHotEncode(label, classes) {
    const encoding = Array(classes.length).fill(0);
    const index = classes.indexOf(label);
    if (index !== -1) encoding[index] = 1;
    return encoding;
}

export async function predictPose(model, keypoints) {
    if (!model || !keypoints || keypoints.length !== 66) {
        return "Unknown";
    }

    const flattenedKeypoints = flattenKeypoints(keypoints);

    if (flattenedKeypoints.length === 0) {
        return "Unknown";
    }

    const max = Math.max(...flattenedKeypoints);
    if (max === 0) {
        return "Unknown";
    }
    const normalizedKeypoints = flattenedKeypoints.map(value => value / max);

    return new Promise(resolve => {
        model.classify(normalizedKeypoints, (err, results) => {
            if (err || !results?.[0]?.label) {
                resolve("Unknown");
            } else {
                resolve(results[0].label);
            }
        });
    });
}