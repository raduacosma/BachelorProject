#include "generateHyperparams.h"
#include "hyperparamSpec.h"
#include <iostream>
#include <random>
#include <string>
#include <filesystem>
#include <regex>
#include <vector>
#include <fstream>

void computeRewardStatistics();
void computeRandomForBest()
{
    std::ifstream in("BESTTHREEHYPERPARAMFILES.txt");
    std::regex firstPartRegex("[0-9]+");
    std::string fileName;
    while(std::getline(in,fileName))
    {
        auto fileBegin =
            std::sregex_iterator(fileName.begin(), fileName.end(), firstPartRegex);
        auto fileEnd = std::sregex_iterator();
        std::string first;
        std::string second;
        for (std::sregex_iterator i = fileBegin; i != fileEnd; ++i) {
            std::smatch match = *i;
            std::string match_str = match.str();
            if(first.empty())
                first = match_str;
            else if(second.empty())
            {
                second = match_str;
            }
            else throw std::runtime_error("matched more than 2 numbers");
        }
        generateHyperparams("",fileName,second+".txt");
    }
}
int main() {
    computeRewardStatistics();
//computeRandomForBest();
//    generateHyperparams("","602seed_maxNrSteps_nrRollouts_000.txt","sarsaOne.txt");
}
void computeRewardStatistics()
{
    std::ofstream out{"finalBestTestToCompare.txt"};
    std::string folderPath = "MONTECARLOFINALRESULTS.txt";
    std::string hyperparamFolderName = "seed_maxNrSteps_nrRollouts";
    std::regex firstPartRegex("[0-9]+");
    size_t resultCnt = 0;
    using infoTuple = std::tuple<std::string,float,std::string,size_t,float,std::string,float,std::string,float,std::string,float,std::string,float,std::string>;
    std::map<std::string,std::map<std::string,infoTuple>> rewardMap;
    for(auto const &file:std::filesystem::recursive_directory_iterator(folderPath))
    {
        std::ifstream in{ file.path() };
        if (in.peek() == std::ifstream::traits_type::eof())
        {
            std::cout<<file.path()<<std::endl;
            throw std::runtime_error("emptyfile");
        }

        std::string hyperparamsPath;
        in >>hyperparamsPath;
        std::string fileName = std::filesystem::path(hyperparamsPath).filename();
        if(fileName.find(hyperparamFolderName) == std::string::npos)
        {
//            std::cout<<file.path()<<std::endl;
            continue;
        }

        auto fileBegin =
            std::sregex_iterator(fileName.begin(), fileName.end(), firstPartRegex);
        auto fileEnd = std::sregex_iterator();
        std::string first;
        std::string second;
        for (std::sregex_iterator i = fileBegin; i != fileEnd; ++i) {
            std::smatch match = *i;
            std::string match_str = match.str();
            if(first.empty())
                first = match_str;
            else if(second.empty())
            {
                second = match_str;
                ++resultCnt;
            }
            else throw std::runtime_error("matched more than 2 numbers");
        }
        std::string resultLabel;
        in>>resultLabel;
        if(resultLabel!="rewardsMeanLast")
            throw std::runtime_error("could not parse reward mean");
        float rewardMeanLast;
        in >> rewardMeanLast;
        auto &item = rewardMap[second][first.substr(1)];
        if(get<0>(item).empty())
            get<0>(item)=fileName;
        get<1>(item)+=rewardMeanLast/8.0f;
        get<2>(item)+=std::to_string(rewardMeanLast)+',';
        ++get<3>(item);
        in >> resultLabel;
        if(resultLabel!="opPredMeanLast")
            throw std::runtime_error("could not find opponent prediction percentage");
        in >> rewardMeanLast;
        get<4>(item)+=rewardMeanLast/8.0f;
        get<5>(item)+=std::to_string(rewardMeanLast)+',';
        in >> resultLabel;
        if(resultLabel!="opponentFoundPredMeanLast")
            throw std::runtime_error("could not find found opponent prediction percentage");
        in >> resultLabel;
        if(resultLabel == "nan" or resultLabel == "-nan")
            rewardMeanLast = 0.0f;
        else
            rewardMeanLast = std::stof(resultLabel);
//        in >> rewardMeanLast;
        get<6>(item)+=rewardMeanLast/8.0f;
        get<7>(item)+=std::to_string(rewardMeanLast)+',';
        in >> resultLabel;
        if(resultLabel!="killedByOpponentMeanLast")
        {
            std::cout<<file.path()<<std::endl;
            std::cout<<resultLabel<<std::endl;
            throw std::runtime_error("could not find killed by opponent mean");
        }

        in >> rewardMeanLast;
        get<8>(item)+=rewardMeanLast/8.0f;
        get<9>(item)+=std::to_string(rewardMeanLast)+',';
        std::string copy1,copy2,copy3;
        in >> resultLabel>>copy1 >> copy2;
        resultLabel+=copy1 + copy2;
        if(resultLabel!="opponentrecognitionpercentage")
        {
            std::cout<<resultLabel<<std::endl;
            std::cout<<file.path()<<std::endl;
            throw std::runtime_error("could not find opponent recognition percentage");
        }
        in >> resultLabel;
        if(resultLabel == "nan" or resultLabel == "-nan")
            rewardMeanLast = 0.0f;
        else
            rewardMeanLast = std::stof(resultLabel);
//        in >> rewardMeanLast;
        get<10>(item)+=rewardMeanLast/8.0f;
        get<11>(item)+=std::to_string(rewardMeanLast)+',';
        in >> resultLabel>>copy1 >> copy2>>copy3;
        resultLabel+=copy1 + copy2+copy3;
        if(resultLabel!="predictednrofopponents")
            throw std::runtime_error("could not find predicted nr of opponents");
        in>>rewardMeanLast;
        get<12>(item)+=rewardMeanLast/8.0f;
        get<13>(item)+=std::to_string(rewardMeanLast)+',';
//        out<<first<<"      "<<second<<'\n';
    }
    for(auto const &item:rewardMap)
    {
//        if(item.first[2]!='0')
//            continue;
        infoTuple maxx;
        get<1>(maxx)=-10000;
        for (auto const &nested : item.second)
        {
//            if(get<3>(nested.second)!=8)
//                continue;
            if(get<1>(nested.second)>get<1>(maxx))
            {
                maxx = nested.second;
            }
        }
        out<<item.first<<":\n";
        out<<"finishedRunNr: "<<get<3>(maxx)<<'\n';
        out<<"maxRewardMean: "<<get<1>(maxx)<<'\n';
        out<<"allRewardMeans\n"<<get<2>(maxx)<<'\n';
        out<<"opponentPredictionPercentage\n"<<get<4>(maxx)<<'\n';
        out<<"listOpponentPredictionPercentage\n"<<get<5>(maxx)<<'\n';
        out<<"foundOpponentPredictionPercentage\n"<<get<6>(maxx)<<'\n';
        out<<"listFoundOpponentPredictionPercentage\n"<<get<7>(maxx)<<'\n';
        out<<"killedByOpponentPercentage\n"<<get<8>(maxx)<<'\n';
        out<<"listKilledByOpponentPercentage\n"<<get<9>(maxx)<<'\n';
        out<<"opponentRecognitionPercentage\n"<<get<10>(maxx)<<'\n';
        out<<"listOpponentRecognitionPercentage\n"<<get<11>(maxx)<<'\n';
        out<<"predicted nr of opponents\n"<<get<12>(maxx)<<'\n';
        out<<"listPredicted nr of opponents\n"<<get<13>(maxx)<<'\n';
        out<<"maxFileName\n"<<get<0>(maxx)<<"\n\n\n\n";
        generateHyperparams(hyperparamFolderName+"/",get<0>(maxx),item.first+".txt");
    }
    //    for(auto const &item:rewardMap)
    //        for(auto const &nested:item.second)
    //        {
    //            out << "begin" << '\n';
    //            out << item.first << '\n'
    //                      << nested.first << '\n'
    //                      << nested.second.first << '\n'
    //                      << nested.second.second << '\n';
    //            out<<"end"<<'\n';
    //        }

    out<< "result count: "<<resultCnt<<std::endl;
}

