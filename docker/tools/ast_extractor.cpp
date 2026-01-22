/**
 * VEGA-Verified AST Extractor
 * 
 * Clang LibTooling-based tool to extract function definitions,
 * switch statements, and control flow from LLVM backend source files.
 * 
 * Compile:
 *   clang++ -std=c++17 ast_extractor.cpp \
 *     $(llvm-config-18 --cxxflags --ldflags --libs all --system-libs) \
 *     -lclang-cpp -o ast_extractor
 * 
 * Usage:
 *   ./ast_extractor <source_file> -- -I/path/to/llvm/include
 */

#include "clang/AST/AST.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <string>
#include <vector>
#include <map>

using namespace clang;
using namespace clang::tooling;

// Command line options
static llvm::cl::OptionCategory ASTExtractorCategory("AST Extractor Options");
static llvm::cl::opt<std::string> OutputFile("o", 
    llvm::cl::desc("Output JSON file"),
    llvm::cl::value_desc("filename"),
    llvm::cl::cat(ASTExtractorCategory));
static llvm::cl::opt<bool> ExtractSwitches("switches",
    llvm::cl::desc("Extract switch statements"),
    llvm::cl::init(true),
    llvm::cl::cat(ASTExtractorCategory));

// Data structures for extracted information
struct ExtractedCase {
    std::string label;
    std::string returnValue;
    int lineNumber;
    bool hasFallthrough;
};

struct SwitchStatement {
    std::string functionName;
    std::string conditionVar;
    std::vector<ExtractedCase> cases;
    std::string defaultCase;
    int lineNumber;
};

struct FunctionInfo {
    std::string name;
    std::string returnType;
    std::vector<std::pair<std::string, std::string>> parameters;
    std::string body;
    int startLine;
    int endLine;
    std::vector<SwitchStatement> switches;
    std::vector<std::string> calledFunctions;
    bool isVirtual;
    bool isConst;
};

// Global storage for extracted data
std::vector<FunctionInfo> ExtractedFunctions;

class ASTExtractorVisitor : public RecursiveASTVisitor<ASTExtractorVisitor> {
public:
    explicit ASTExtractorVisitor(ASTContext *Context) : Context(Context) {}

    bool VisitFunctionDecl(FunctionDecl *FD) {
        // Skip non-definition declarations
        if (!FD->hasBody())
            return true;
        
        // Get source location
        SourceManager &SM = Context->getSourceManager();
        SourceLocation Loc = FD->getLocation();
        
        // Skip system headers
        if (SM.isInSystemHeader(Loc))
            return true;
        
        FunctionInfo Info;
        Info.name = FD->getNameAsString();
        Info.returnType = FD->getReturnType().getAsString();
        Info.startLine = SM.getSpellingLineNumber(FD->getBeginLoc());
        Info.endLine = SM.getSpellingLineNumber(FD->getEndLoc());
        Info.isVirtual = FD->isVirtualAsWritten();
        
        // Get const qualifier for methods
        if (auto *MD = dyn_cast<CXXMethodDecl>(FD)) {
            Info.isConst = MD->isConst();
        }
        
        // Extract parameters
        for (const auto *Param : FD->parameters()) {
            Info.parameters.push_back({
                Param->getType().getAsString(),
                Param->getNameAsString()
            });
        }
        
        // Extract body text
        if (FD->hasBody()) {
            Stmt *Body = FD->getBody();
            SourceRange Range = Body->getSourceRange();
            CharSourceRange CharRange = CharSourceRange::getTokenRange(Range);
            Info.body = Lexer::getSourceText(CharRange, SM, Context->getLangOpts()).str();
        }
        
        // Visit body for switches and calls
        if (FD->hasBody()) {
            CurrentFunction = &Info;
            TraverseStmt(FD->getBody());
            CurrentFunction = nullptr;
        }
        
        ExtractedFunctions.push_back(Info);
        return true;
    }

    bool VisitSwitchStmt(SwitchStmt *SS) {
        if (!CurrentFunction || !ExtractSwitches)
            return true;
        
        SwitchStatement Switch;
        SourceManager &SM = Context->getSourceManager();
        Switch.lineNumber = SM.getSpellingLineNumber(SS->getBeginLoc());
        Switch.functionName = CurrentFunction->name;
        
        // Get condition variable
        if (Expr *Cond = SS->getCond()) {
            if (auto *DRE = dyn_cast<DeclRefExpr>(Cond->IgnoreParenCasts())) {
                Switch.conditionVar = DRE->getNameInfo().getAsString();
            } else {
                // Get the condition expression as string
                CharSourceRange Range = CharSourceRange::getTokenRange(Cond->getSourceRange());
                Switch.conditionVar = Lexer::getSourceText(Range, SM, Context->getLangOpts()).str();
            }
        }
        
        // Extract cases
        if (clang::SwitchCase *SC = SS->getSwitchCaseList()) {
            while (SC) {
                ExtractedCase CaseInfo;
                CaseInfo.lineNumber = SM.getSpellingLineNumber(SC->getBeginLoc());
                CaseInfo.hasFallthrough = false;
                
                if (auto *CS = dyn_cast<CaseStmt>(SC)) {
                    // Get case label
                    if (Expr *LHS = CS->getLHS()) {
                        CharSourceRange Range = CharSourceRange::getTokenRange(LHS->getSourceRange());
                        CaseInfo.label = Lexer::getSourceText(Range, SM, Context->getLangOpts()).str();
                    }
                    
                    // Get return value or body
                    if (Stmt *Sub = CS->getSubStmt()) {
                        ExtractCaseBody(Sub, CaseInfo, SM);
                    }
                } else if (auto *DS = dyn_cast<DefaultStmt>(SC)) {
                    CaseInfo.label = "default";
                    if (Stmt *Sub = DS->getSubStmt()) {
                        ExtractCaseBody(Sub, CaseInfo, SM);
                    }
                    Switch.defaultCase = CaseInfo.returnValue;
                }
                
                if (CaseInfo.label != "default") {
                    Switch.cases.push_back(CaseInfo);
                }
                
                SC = SC->getNextSwitchCase();
            }
        }
        
        CurrentFunction->switches.push_back(Switch);
        return true;
    }

    bool VisitCallExpr(CallExpr *CE) {
        if (!CurrentFunction)
            return true;
        
        if (FunctionDecl *Callee = CE->getDirectCallee()) {
            CurrentFunction->calledFunctions.push_back(Callee->getNameAsString());
        }
        return true;
    }

private:
    ASTContext *Context;
    FunctionInfo *CurrentFunction = nullptr;

    void ExtractCaseBody(Stmt *Body, ExtractedCase &CaseInfo, SourceManager &SM) {
        // Check for return statement
        if (auto *RS = dyn_cast<ReturnStmt>(Body)) {
            if (Expr *RetVal = RS->getRetValue()) {
                CharSourceRange Range = CharSourceRange::getTokenRange(RetVal->getSourceRange());
                CaseInfo.returnValue = Lexer::getSourceText(Range, SM, Context->getLangOpts()).str();
            }
        } else if (auto *CS = dyn_cast<CompoundStmt>(Body)) {
            // Look for return in compound statement
            for (auto *S : CS->body()) {
                if (auto *RS = dyn_cast<ReturnStmt>(S)) {
                    if (Expr *RetVal = RS->getRetValue()) {
                        CharSourceRange Range = CharSourceRange::getTokenRange(RetVal->getSourceRange());
                        CaseInfo.returnValue = Lexer::getSourceText(Range, SM, Context->getLangOpts()).str();
                    }
                    break;
                }
                // Check for break
                if (isa<BreakStmt>(S)) {
                    break;
                }
            }
            // No break found = fallthrough
            bool hasBreakOrReturn = false;
            for (auto *S : CS->body()) {
                if (isa<BreakStmt>(S) || isa<ReturnStmt>(S)) {
                    hasBreakOrReturn = true;
                    break;
                }
            }
            CaseInfo.hasFallthrough = !hasBreakOrReturn;
        } else if (isa<CaseStmt>(Body)) {
            // Fallthrough to next case
            CaseInfo.hasFallthrough = true;
        }
    }
};

class ASTExtractorConsumer : public ASTConsumer {
public:
    explicit ASTExtractorConsumer(ASTContext *Context) : Visitor(Context) {}

    void HandleTranslationUnit(ASTContext &Context) override {
        Visitor.TraverseDecl(Context.getTranslationUnitDecl());
    }

private:
    ASTExtractorVisitor Visitor;
};

class ASTExtractorAction : public ASTFrontendAction {
public:
    std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI,
                                                   StringRef InFile) override {
        return std::make_unique<ASTExtractorConsumer>(&CI.getASTContext());
    }
};

void OutputJSON() {
    llvm::json::Array Functions;
    
    for (const auto &FI : ExtractedFunctions) {
        llvm::json::Object Func;
        Func["name"] = FI.name;
        Func["return_type"] = FI.returnType;
        Func["start_line"] = FI.startLine;
        Func["end_line"] = FI.endLine;
        Func["is_virtual"] = FI.isVirtual;
        Func["is_const"] = FI.isConst;
        
        // Parameters
        llvm::json::Array Params;
        for (const auto &P : FI.parameters) {
            llvm::json::Object Param;
            Param["type"] = P.first;
            Param["name"] = P.second;
            Params.push_back(std::move(Param));
        }
        Func["parameters"] = std::move(Params);
        
        // Switches
        llvm::json::Array Switches;
        for (const auto &SW : FI.switches) {
            llvm::json::Object Switch;
            Switch["function"] = SW.functionName;
            Switch["condition"] = SW.conditionVar;
            Switch["line"] = SW.lineNumber;
            Switch["default"] = SW.defaultCase;
            
            llvm::json::Array Cases;
            for (const auto &C : SW.cases) {
                llvm::json::Object Case;
                Case["label"] = C.label;
                Case["return"] = C.returnValue;
                Case["line"] = C.lineNumber;
                Case["fallthrough"] = C.hasFallthrough;
                Cases.push_back(std::move(Case));
            }
            Switch["cases"] = std::move(Cases);
            Switches.push_back(std::move(Switch));
        }
        Func["switches"] = std::move(Switches);
        
        // Called functions
        llvm::json::Array Calls;
        for (const auto &C : FI.calledFunctions) {
            Calls.push_back(C);
        }
        Func["calls"] = std::move(Calls);
        
        // Body (truncated for large functions)
        if (FI.body.size() > 10000) {
            Func["body"] = FI.body.substr(0, 10000) + "... (truncated)";
        } else {
            Func["body"] = FI.body;
        }
        
        Functions.push_back(std::move(Func));
    }
    
    llvm::json::Object Root;
    Root["version"] = "1.0";
    Root["tool"] = "vega-ast-extractor";
    Root["function_count"] = (int)ExtractedFunctions.size();
    Root["functions"] = std::move(Functions);
    
    // Output
    std::error_code EC;
    std::string OutFileName = OutputFile.empty() ? "-" : OutputFile.getValue();
    llvm::raw_fd_ostream OS(OutFileName, EC);
    if (EC) {
        llvm::errs() << "Error opening output file: " << EC.message() << "\n";
        return;
    }
    
    OS << llvm::json::Value(std::move(Root));
}

int main(int argc, const char **argv) {
    auto ExpectedParser = CommonOptionsParser::create(argc, argv, ASTExtractorCategory);
    if (!ExpectedParser) {
        llvm::errs() << ExpectedParser.takeError();
        return 1;
    }
    CommonOptionsParser &OptionsParser = ExpectedParser.get();
    
    ClangTool Tool(OptionsParser.getCompilations(),
                   OptionsParser.getSourcePathList());
    
    int Result = Tool.run(newFrontendActionFactory<ASTExtractorAction>().get());
    
    if (Result == 0) {
        OutputJSON();
    }
    
    return Result;
}
