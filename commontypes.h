#ifndef COMMONTYPES_H
#define COMMONTYPES_H

#include <QString>
#include <QJsonObject>
#include <QJsonArray>

/**
 * @file commontypes.h
 * @brief Common data structures and types used throughout NeuroForge application
 */

/**
 * @brief Parameters for incremental/rolling horizon training
 */
struct IncrementalTrainingParams {
    double windowSize = 200.0;           ///< Size of each training window
    double windowStep = 200.0;           ///< Step size between windows (overlap if < windowSize)
    int epochsPerWindow = 50;            ///< Number of epochs to train on each window
    int batchSize = 32;                  ///< Batch size for training
    double learningRate = 0.001;         ///< Learning rate
    bool useOverlap = false;             ///< Whether windows overlap
    bool resetOnNewWindow = false;       ///< Whether to reset optimizer state on new window

    IncrementalTrainingParams() = default;

    QJsonObject toJson() const {
        QJsonObject obj;
        obj["windowSize"] = windowSize;
        obj["windowStep"] = windowStep;
        obj["epochsPerWindow"] = epochsPerWindow;
        obj["batchSize"] = batchSize;
        obj["learningRate"] = learningRate;
        obj["useOverlap"] = useOverlap;
        obj["resetOnNewWindow"] = resetOnNewWindow;
        return obj;
    }

    void fromJson(const QJsonObject& obj) {
        windowSize = obj["windowSize"].toDouble(200.0);
        windowStep = obj["windowStep"].toDouble(200.0);
        epochsPerWindow = obj["epochsPerWindow"].toInt(50);
        batchSize = obj["batchSize"].toInt(32);
        learningRate = obj["learningRate"].toDouble(0.001);
        useOverlap = obj["useOverlap"].toBool(false);
        resetOnNewWindow = obj["resetOnNewWindow"].toBool(false);
    }
};

/**
 * @brief Neural network architecture configuration
 */
struct NetworkArchitecture {
    std::vector<int> hiddenLayers;           ///< Number of nodes in each hidden layer
    std::vector<std::string> activations;    ///< Activation function for each layer
    std::string outputActivation = "linear"; ///< Output layer activation
    std::vector<std::vector<int>> lags;      ///< Lag configuration for each input series
    bool isConfigured = false;               ///< Whether architecture has been set

    NetworkArchitecture() = default;

    QJsonObject toJson() const {
        QJsonObject obj;
        obj["isConfigured"] = isConfigured;
        obj["outputActivation"] = QString::fromStdString(outputActivation);

        // Hidden layers
        QJsonArray layersArray;
        for (int size : hiddenLayers) {
            layersArray.append(size);
        }
        obj["hiddenLayers"] = layersArray;

        // Activations
        QJsonArray activationsArray;
        for (const auto& act : activations) {
            activationsArray.append(QString::fromStdString(act));
        }
        obj["activations"] = activationsArray;

        // Lags
        QJsonArray lagsArray;
        for (const auto& lagVec : lags) {
            QJsonArray innerArray;
            for (int lag : lagVec) {
                innerArray.append(lag);
            }
            lagsArray.append(innerArray);
        }
        obj["lags"] = lagsArray;

        return obj;
    }

    void fromJson(const QJsonObject& obj) {
        isConfigured = obj["isConfigured"].toBool(false);
        outputActivation = obj["outputActivation"].toString("linear").toStdString();

        // Hidden layers
        hiddenLayers.clear();
        QJsonArray layersArray = obj["hiddenLayers"].toArray();
        for (const auto& val : layersArray) {
            hiddenLayers.push_back(val.toInt());
        }

        // Activations
        activations.clear();
        QJsonArray activationsArray = obj["activations"].toArray();
        for (const auto& val : activationsArray) {
            activations.push_back(val.toString().toStdString());
        }

        // Lags
        lags.clear();
        QJsonArray lagsArray = obj["lags"].toArray();
        for (const auto& innerVal : lagsArray) {
            QJsonArray innerArray = innerVal.toArray();
            std::vector<int> lagVec;
            for (const auto& lag : innerArray) {
                lagVec.push_back(lag.toInt());
            }
            lags.push_back(lagVec);
        }
    }
};

/**
 * @brief Project configuration for NeuroForge
 */
struct ProjectConfig {
    QString projectName;
    QString inputDataPath;
    QString targetDataPath;

    // GA Configuration
    int gaPopulation = 100;
    int gaGenerations = 100;
    double gaMutationRate = 0.01;
    int gaEliteCount = 2;
    QString gaOutputPath = "./";


    // Data configuration
    double splitRatio = 0.7;
    double tStart = 0.0;
    double tEnd = 100.0;
    double dt = 0.1;

    // Network architecture
    NetworkArchitecture networkArchitecture;

    // Incremental training parameters
    IncrementalTrainingParams incrementalParams;

    ProjectConfig() = default;

    QJsonObject toJson() const {
        QJsonObject obj;
        obj["projectName"] = projectName;
        obj["inputDataPath"] = inputDataPath;
        obj["targetDataPath"] = targetDataPath;

        // GA settings
        QJsonObject gaObj;
        gaObj["population"] = gaPopulation;
        gaObj["generations"] = gaGenerations;
        gaObj["mutationRate"] = gaMutationRate;
        gaObj["eliteCount"] = gaEliteCount;
        obj["gaSettings"] = gaObj;

        // Data settings
        QJsonObject dataObj;
        dataObj["splitRatio"] = splitRatio;
        dataObj["tStart"] = tStart;
        dataObj["tEnd"] = tEnd;
        dataObj["dt"] = dt;
        obj["dataSettings"] = dataObj;

        // Network architecture
        obj["networkArchitecture"] = networkArchitecture.toJson();

        // Incremental training
        obj["incrementalTraining"] = incrementalParams.toJson();

        return obj;
    }

    void fromJson(const QJsonObject& obj) {
        projectName = obj["projectName"].toString("");
        inputDataPath = obj["inputDataPath"].toString("");
        targetDataPath = obj["targetDataPath"].toString("");

        // GA settings
        if (obj.contains("gaSettings")) {
            QJsonObject gaObj = obj["gaSettings"].toObject();
            gaPopulation = gaObj["population"].toInt(100);
            gaGenerations = gaObj["generations"].toInt(100);
            gaMutationRate = gaObj["mutationRate"].toDouble(0.01);
            gaEliteCount = gaObj["eliteCount"].toInt(2);
            gaOutputPath = gaObj["outputPath"].toString("./");
        }

        // Data settings
        if (obj.contains("dataSettings")) {
            QJsonObject dataObj = obj["dataSettings"].toObject();
            splitRatio = dataObj["splitRatio"].toDouble(0.7);
            tStart = dataObj["tStart"].toDouble(0.0);
            tEnd = dataObj["tEnd"].toDouble(100.0);
            dt = dataObj["dt"].toDouble(0.1);
        }

        // Network architecture
        if (obj.contains("networkArchitecture")) {
            networkArchitecture.fromJson(obj["networkArchitecture"].toObject());
        }

        // Incremental training
        if (obj.contains("incrementalTraining")) {
            incrementalParams.fromJson(obj["incrementalTraining"].toObject());
        }
    }
};
#endif // COMMONTYPES_H
