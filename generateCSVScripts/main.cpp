#include <iostream>
#include <fstream>
#include <regex>
#include <filesystem>
#include <map>
void generateSeedPairings()
{
    std::string folderPath = "TruncComplexResults";
    std::string copyFolderPath = "ComplexFinal";
    std::regex firstPartRegex("[0-9]+");
    size_t maxSize = 0;
    std::map<std::string,std::vector<std::vector<std::string>>> pairedMap;
    std::map<std::string,size_t> maxSizes;
    for(auto const &file:std::filesystem::recursive_directory_iterator(folderPath))
    {
        std::string fileName = file.path().filename();
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
        pairedMap[second].push_back(std::vector<std::string>());
        std::ifstream in(file.path());
        std::string currentLine;
        in>>currentLine;
        size_t sz = pairedMap[second].size();
        std::string i = std::to_string(sz);
        if(currentLine == "actualOpType,predOpType,rewards,opPredPerc,foundOpPredPerc,killedByOpPerc")
            pairedMap[second].back().push_back("actualOpType"+i+",predOpType"+i+",rewards"+i+",opPredPerc"+i+",foundOpPredPerc"+i+",killedByOpPerc"+i);
//        if(sz!=8)
//            pairedMap[second].back().back()+=',';
        while (std::getline(in,currentLine)) {
            if(currentLine.empty())
                continue;
            pairedMap[second].back().push_back(currentLine);
        }
        if(maxSizes[second]<pairedMap[second].back().size())
        {
            maxSizes[second]=pairedMap[second].back().size();
        }
    }
    for(auto const &item:pairedMap)
    {
        std::ofstream out(copyFolderPath+'/'+item.first+".txt");
        for(size_t idx=0;idx!=maxSizes[item.first];++idx)
        {
            for(size_t snd=0;snd!=item.second.size();++snd)
            {
                if(idx>=item.second[snd].size()) {
                    out << "NA,NA,NA,NA,NA,NA";
                    if (snd != item.second.size() - 1)
                        out << ',';
                    continue;
                }
                out<<item.second[snd][idx];
                if(snd!=item.second.size()-1)
                    out<<',';
            }
            out<<'\n';
        }
    }
}
void copyFilesWithoutEnd()
{
    std::string folderPath = "ComplexResults";
    std::string copyFolderPath = "TruncComplexResults";
    std::regex firstPartRegex("[0-9]+");
    size_t resultCnt = 0;
    for(auto const &file:std::filesystem::recursive_directory_iterator(folderPath))
    {
        std::ifstream in{ file.path() };
        if (in.peek() == std::ifstream::traits_type::eof())
        {
            std::cout<<file.path()<<std::endl;
            throw std::runtime_error("emptyfile");
        }

        std::string columnNames;
        in >>columnNames;
        if(columnNames!="actualOpType,predOpType,rewards,opPredPerc,foundOpPredPerc,killedByOpPerc")
            continue;
        std::vector<std::string> csvLines;
        csvLines.push_back(columnNames);
        std::string currentLine;
        std::string copyFileName;
        while (std::getline(in,currentLine))
        {
            if(currentLine == "opponent recognition percentage")
            {
                std::string nvm;
                std::getline(in,nvm);
                std::getline(in,nvm);
                std::getline(in,nvm);
                std::getline(in,copyFileName);
                copyFileName = std::filesystem::path(copyFileName).filename();
                break;
            }
            else
            {
                csvLines.push_back(currentLine);
            }
        }
        std::ofstream copyFileOf(copyFolderPath+'/'+copyFileName);
        for(auto const &line:csvLines)
        {
            copyFileOf<<line<<'\n';
        }
    }
}
int main() {
generateSeedPairings();
    return 0;
}
