import 'package:flutter/material.dart';

class TennisMap extends StatefulWidget {
  const TennisMap({super.key});

  @override
  State<TennisMap> createState() => _TennisMapState();
}

class _TennisMapState extends State<TennisMap> {
  final List<String> entries = <String>['a','b','c','d'];
  final List<int> colorCodes = <int>[600,500,100,200];

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        Container(
          height: 500,
          width: 500,
          margin: EdgeInsets.all(10),
          decoration: BoxDecoration(border: Border.all(color: Colors.black, width: 1)),
          child: Center(child: Text("지도")),
        ),
        Expanded(
          child: ListView.separated(
            padding:const EdgeInsets.all(8),
            itemCount:entries.length,
            itemBuilder: (BuildContext context, int index) => Container(
              height: 50,
              color: Colors.amber[colorCodes[index]],
              child:Center(
                child:Text('테니스장 ${entries[index]}')
              )
            ),
            separatorBuilder: (BuildContext context, int index) => const Divider(),
          ),
        )
      ],
    );
  }
}
