import 'package:flutter/material.dart';

class SwingSimilarity extends StatefulWidget {
  const SwingSimilarity({super.key});

  @override
  State<SwingSimilarity> createState() => _SwingSimilarityState();
}

class _SwingSimilarityState extends State<SwingSimilarity> {
  List<String> playerName = ['로저 페더러', '노박 조코비치', '라파엘 나달', '로드 레이버', '피트 샘프러스', '비욘 보리'];

  @override
  Widget build(BuildContext context) {
    return GridView.builder(
      gridDelegate: SliverGridDelegateWithFixedCrossAxisCount(crossAxisCount: 2),
      itemCount: 6,
      itemBuilder: (context, index) => Container(
        margin: EdgeInsets.all(30),
        width: 100,
        height: 100,
        decoration: BoxDecoration(border: Border.all(color: Colors.black, width: 1)),
        child: Center(
          child: Text(playerName[index]),
        ),
      ),
    );
  }
}

