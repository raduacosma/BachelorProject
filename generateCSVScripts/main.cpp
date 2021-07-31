#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <regex>

// This function manipulates the output of the per-episode statistics
// such that the results of a configuration are all stored in the same file
// so that files from different random seeds are put into one for data processing
void generateSeedPairings()
{
    // folder from which to get files
    std::string folderPath = "TruncComplexResults";
    // folder in which to write new files
    std::string copyFolderPath = "ComplexFinal";
    std::regex firstPartRegex("[0-9]+");
    size_t maxSize = 0;
    std::map<std::string, std::vector<std::vector<std::string>>> pairedMap;
    std::map<std::string, size_t> maxSizes;
    for (auto const &file : std::filesystem::recursive_directory_iterator(folderPath))
    {
        std::string fileName = file.path().filename();
        auto fileBegin = std::sregex_iterator(fileName.begin(), fileName.end(), firstPartRegex);
        auto fileEnd = std::sregex_iterator();
        std::string first;
        std::string second;
        for (std::sregex_iterator i = fileBegin; i != fileEnd; ++i)
        {
            std::smatch match = *i;
            std::string match_str = match.str();
            if (first.empty())
                first = match_str;
            else if (second.empty())
            {
                second = match_str;
            }
            else
                throw std::runtime_error("matched more than 2 numbers");
        }
        pairedMap[second].push_back(std::vector<std::string>());
        std::ifstream in(file.path());
        std::string currentLine;
        in >> currentLine;
        size_t sz = pairedMap[second].size();
        std::string i = std::to_string(sz);
        if (currentLine == "actualOpType,predOpType,rewards,opPredPerc,foundOpPredPerc,killedByOpPerc")
            pairedMap[second].back().push_back("actualOpType" + i + ",predOpType" + i + ",rewards" + i + ",opPredPerc" +
                                               i + ",foundOpPredPerc" + i + ",killedByOpPerc" + i);
        //        if(sz!=8)
        //            pairedMap[second].back().back()+=',';
        while (std::getline(in, currentLine))
        {
            if (currentLine.empty())
                continue;
            pairedMap[second].back().push_back(currentLine);
        }
        if (maxSizes[second] < pairedMap[second].back().size())
        {
            maxSizes[second] = pairedMap[second].back().size();
        }
    }
    for (auto const &item : pairedMap)
    {
        std::ofstream out(copyFolderPath + '/' + item.first + ".txt");
        for (size_t idx = 0; idx != maxSizes[item.first]; ++idx)
        {
            for (size_t snd = 0; snd != item.second.size(); ++snd)
            {
                if (idx >= item.second[snd].size())
                {
                    out << "NA,NA,NA,NA,NA,NA";
                    if (snd != item.second.size() - 1)
                        out << ',';
                    continue;
                }
                out << item.second[snd][idx];
                if (snd != item.second.size() - 1)
                    out << ',';
            }
            out << '\n';
        }
    }
}

// This function preprocesses the specific output that was outputted by the Peregrine compute cluster
// at the University of Groningen to feed into the generateSeedPairings() function above
void copyFilesWithoutEnd()
{
    // folder from which to get files
    std::string folderPath = "ComplexResults";
    // folder in which to write new files
    std::string copyFolderPath = "TruncComplexResults";
    std::regex firstPartRegex("[0-9]+");
    size_t resultCnt = 0;
    for (auto const &file : std::filesystem::recursive_directory_iterator(folderPath))
    {
        std::ifstream in{ file.path() };
        if (in.peek() == std::ifstream::traits_type::eof())
        {
            std::cout << file.path() << std::endl;
            throw std::runtime_error("emptyfile");
        }

        std::string columnNames;
        in >> columnNames;
        if (columnNames != "actualOpType,predOpType,rewards,opPredPerc,foundOpPredPerc,killedByOpPerc")
            continue;
        std::vector<std::string> csvLines;
        csvLines.push_back(columnNames);
        std::string currentLine;
        std::string copyFileName;
        while (std::getline(in, currentLine))
        {
            if (currentLine == "opponent recognition percentage")
            {
                std::string nvm;
                std::getline(in, nvm);
                std::getline(in, nvm);
                std::getline(in, nvm);
                std::getline(in, copyFileName);
                copyFileName = std::filesystem::path(copyFileName).filename();
                break;
            }
            else
            {
                csvLines.push_back(currentLine);
            }
        }
        std::ofstream copyFileOf(copyFolderPath + '/' + copyFileName);
        for (auto const &line : csvLines)
        {
            copyFileOf << line << '\n';
        }
    }
}
int main()
{
    // first use the copyFilesWithoutEnd() and then generateSeedPairings()
    // if you need to preprocess the output in this way
    generateSeedPairings();
    return 0;
}
